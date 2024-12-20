import copy
import gc
import json
import math
import os
import pickle
import random
from itertools import repeat
from typing import List, Tuple, Union

import faiss
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from data.datasets.base_datasets import TrainingTuple, get_pointcloud_loader
from utils.misc_utils import collate_fn
from utils.o3d_utils import get_matching_indices, make_open3d_point_cloud


def read_json_file(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


class SejongSouthbayLoader(Dataset):  # 定义一个继承自 Dataset 的类 SejongSouthbayLoader
    def __init__(
        self, all_file_loc=None, pcl_transform=None
    ):  # 初始化方法，接受文件路径列表和点云变换函数作为参数
        if all_file_loc == None:  # 如果文件路径列表为 None
            with open(
                "/home/ljc/Dataset/Apollo-Southbay/train_southbay_2_10.pickle", "rb"
            ) as file:  # 打开 Southbay 数据集文件
                self.southbay_data_dict = pickle.load(file)  # 加载 Southbay 数据集字典
            with open(
                "/home/ljc/Dataset/Mulran/Sejong/train_Sejong1_Sejong2_2_10.pickle",
                "rb",
            ) as file:  # 打开 Sejong 数据集文件
                self.sejong_data_dict = pickle.load(file)  # 加载 Sejong 数据集字典

            self.all_file_loc = []  # 初始化文件路径列表
            self.pos_pairs_ind = []  # 初始化正样本对索引列表
            self.non_negative_pairs_ind = []  # 初始化非负样本对索引列表
            self.transforms = []  # 初始化变换矩阵列表
            self.is_southbay = []  # 初始化数据集标识列表

            ############# For Southbay ##################################
            for i in range(len(self.southbay_data_dict)):  # 遍历 Southbay 数据字典
                self.all_file_loc.append(
                    self.southbay_data_dict[i].rel_scan_filepath
                )  # 添加文件路径
                self.transforms.append(self.southbay_data_dict[i].pose)  # 添加变换矩阵
                self.pos_pairs_ind.append(
                    self.southbay_data_dict[i].positives.tolist()
                )  # 添加正样本对索引
                self.non_negative_pairs_ind.append(
                    self.southbay_data_dict[i].non_negatives.tolist()
                )  # 添加非负样本对索引
                self.is_southbay.append(1)  # 标记为 Southbay 数据

            ############# For Sejong ####################################
            len_southbay = len(self.southbay_data_dict)  # 获取 Southbay 数据的长度

            for i in range(len(self.sejong_data_dict)):  # 遍历 Sejong 数据字典
                self.all_file_loc.append(
                    self.sejong_data_dict[i].rel_scan_filepath
                )  # 添加文件路径
                self.transforms.append(self.sejong_data_dict[i].pose)  # 添加变换矩阵
                pos_pairs = (
                    np.array(self.sejong_data_dict[i].positives) + len_southbay
                ).tolist()  # 计算正样本对索引
                self.pos_pairs_ind.append(pos_pairs)  # 添加正样本对索引
                non_neg_pairs = (
                    np.array(self.sejong_data_dict[i].non_negatives) + len_southbay
                ).tolist()  # 计算非负样本对索引
                self.non_negative_pairs_ind.append(non_neg_pairs)  # 添加非负样本对索引
                self.is_southbay.append(0)  # 标记为 Sejong 数据

            self.transforms = np.array(self.transforms)  # 将变换矩阵列表转换为 numpy 数组
        else:
            self.all_file_loc = all_file_loc  # 如果提供了文件路径列表，则直接使用

        self.voxel_size = 0.5  # 设置体素大小
        self.pcl_transform = pcl_transform  # 设置点云变换函数
        self.mulran_pc_loader = get_pointcloud_loader("mulran")  # 获取 Mulran 点云加载器
        self.southbay_pc_loader = get_pointcloud_loader("southbay")  # 获取 Southbay 点云加载器

    def data_prepare(
        self, xyzr, voxel_size=np.array([0.1, 0.1, 0.1])
    ):  # 数据准备方法，接受点云数据和体素大小作为参数
        lidar_pc = copy.deepcopy(xyzr)  # 深拷贝点云数据
        coords = np.round(lidar_pc[:, :3] / voxel_size)  # 计算体素化后的坐标
        coords_min = coords.min(0, keepdims=1)  # 计算坐标的最小值
        coords -= coords_min  # 坐标减去最小值
        feats = lidar_pc  # 特征等于点云数据

        hash_vals, _, uniq_idx = self.sparse_quantize(
            coords, return_index=True, return_hash=True
        )  # 稀疏量化坐标，返回哈希值和唯一索引
        coord_voxel, feat = coords[uniq_idx], feats[uniq_idx]  # 获取体素化后的坐标和特征
        coord = copy.deepcopy(feat[:, :3])  # 深拷贝特征的前三列作为坐标

        coord = torch.FloatTensor(coord)  # 将坐标转换为 FloatTensor
        feat = torch.FloatTensor(feat)  # 将特征转换为 FloatTensor
        coord_voxel = torch.LongTensor(coord_voxel)  # 将体素化后的坐标转换为 LongTensor
        return coord_voxel, coord, feat  # 返回体素化后的坐标、原始坐标和特征

    def sparse_quantize(
        self,
        coords,
        voxel_size=1,
        return_index=False,
        return_inverse=False,
        return_hash=False,
    ):  # 稀疏量化方法
        if isinstance(voxel_size, (float, int)):  # 如果体素大小是浮点数或整数
            voxel_size = tuple(repeat(voxel_size, 3))  # 将体素大小重复三次，形成元组
        assert (
            isinstance(voxel_size, tuple) and len(voxel_size) == 3
        )  # 断言体素大小是长度为 3 的元组

        voxel_size = np.array(voxel_size)  # 将体素大小转换为 numpy 数组
        coords = np.floor(coords / voxel_size).astype(
            np.int32
        )  # 计算体素化后的坐标，并转换为 int32 类型

        hash_vals, indices, inverse_indices = np.unique(
            self.ravel_hash(coords), return_index=True, return_inverse=True
        )  # 计算唯一哈希值和索引
        coords = coords[indices]  # 获取唯一索引对应的坐标

        if return_hash:
            outputs = [hash_vals, coords]  # 如果返回哈希值，则输出哈希值和坐标
        else:
            outputs = [coords]  # 否则只输出坐标

        if return_index:  # 如果返回索引
            outputs += [indices]  # 添加索引到输出
        if return_inverse:  # 如果返回逆索引
            outputs += [inverse_indices]  # 添加逆索引到输出
        return (
            outputs[0] if len(outputs) == 1 else outputs
        )  # 如果输出长度为 1，则返回第一个元素，否则返回输出列表

    def ravel_hash(self, x: np.ndarray) -> np.ndarray:  # 哈希方法，接受二维数组 x 作为参数
        assert x.ndim == 2, x.shape  # 断言 x 是二维数组

        x -= np.min(x, axis=0)  # 坐标减去最小值
        x = x.astype(np.uint64, copy=False)  # 转换为 uint64 类型
        xmax = np.max(x, axis=0).astype(np.uint64) + 1  # 计算坐标的最大值加 1

        h = np.zeros(x.shape[0], dtype=np.uint64)  # 初始化哈希值数组
        for k in range(x.shape[1] - 1):  # 遍历坐标的每一列
            h += x[:, k]  # 累加坐标值
            h *= xmax[k + 1]  # 乘以下一列的最大值
        h += x[:, -1]  # 累加最后一列的坐标值
        return h  # 返回哈希值

    def __len__(self):  # 数据集长度方法
        return len(self.all_file_loc)  # 返回文件路径列表的长度

    def read_pcd_file(self, filename):  # 读取点云文件方法
        if filename[22] == "A":  # 如果文件名的第 23 个字符是 'A'
            xyzr = self.southbay_pc_loader(filename)  # 使用 Southbay 加载器读取点云数据
        else:
            xyzr = self.mulran_pc_loader(filename)  # 否则使用 Mulran 加载器读取点云数据
        return xyzr  # 返回点云数据

    def __getitem__(self, idx):  # 获取数据方法
        filename = self.all_file_loc[idx]  # 根据索引获取文件路径
        xyzr = self.read_pcd_file(filename)  # 读取点云数据
        if self.pcl_transform is not None:  # 如果有点云变换函数
            xyzr = self.pcl_transform(xyzr)  # 对点云数据进行变换
        if len(xyzr) > 0:  # 如果点云数据不为空
            coords, xyz, feats = self.data_prepare(
                xyzr,
                voxel_size=np.array(
                    [self.voxel_size, self.voxel_size, self.voxel_size]
                ),
            )  # 准备数据
        else:
            coords = torch.FloatTensor(np.ones([100, 3]) * (-1))  # 否则初始化坐标为 -1
            xyz = torch.FloatTensor(np.ones([100, 3]) * (-1))  # 初始化原始坐标为 -1
            feats = torch.FloatTensor(np.ones([100, 3]) * (-1))  # 初始化特征为 -1
        return coords, xyz, feats  # 返回体素化后的坐标、原始坐标和特征


class SejongSouthbayTupleLoader(
    SejongSouthbayLoader
):  # 定义一个继承自 SejongSouthbayLoader 的类 SejongSouthbayTupleLoader
    def __init__(
        self, cached_queries=1000, pcl_transform=None
    ):  # 初始化方法，接受缓存查询数量和点云变换函数作为参数
        super().__init__(pcl_transform=pcl_transform)  # 调用父类的初始化方法
        self.cached_queries = cached_queries  # 设置缓存查询数量
        self.nNeg = 5  # 设置负样本数量
        self.margin = 0.1  # 设置损失函数的边距
        self.prev_epoch_neg = (
            -np.ones((len(self.all_file_loc), self.nNeg), dtype=int)
        ).tolist()  # 初始化上一轮训练的负样本索引

    def new_epoch(self):  # 新的训练轮次方法

        # 计算每轮训练需要的子缓存数量
        self.nCacheSubset = math.ceil(len(self.all_file_loc) / self.cached_queries)

        # 获取所有索引
        arr = np.arange(len(self.all_file_loc))

        arr = np.random.permutation(arr)  # 随机打乱索引

        # 计算子缓存索引
        self.subcache_indices = np.array_split(arr, self.nCacheSubset)

    def update_subcache(self, net, outputdim):  # 更新子缓存方法

        # 重置三元组
        self.triplets = []

        # 获取当前子缓存的查询索引
        qidxs = np.asarray(self.subcache_indices[self.current_subset])

        # 获取对应的正样本索引
        pos_samples = []
        queries = []
        neg_samples = []

        pidxs = []
        all_pidxs = []
        k = 0
        for i in qidxs:
            queries.append(self.all_file_loc[i])  # 添加查询文件路径

            all_pidxs.extend(self.pos_pairs_ind[i])  # 添加所有正样本索引
            pidx = np.random.choice(self.pos_pairs_ind[i])  # 随机选择一个正样本索引
            pidxs.append(pidx)  # 添加正样本索引
            k = k + 1
        pidxs = np.unique(np.array(pidxs))  # 获取唯一的正样本索引

        for pidx in pidxs:
            pos_samples.append(self.all_file_loc[pidx])  # 添加正样本文件路径

        all_pidxs = np.unique(np.array(all_pidxs)).tolist()  # 获取唯一的所有正样本索引

        set1 = set(np.arange(len(self.all_file_loc)).tolist())  # 获取所有文件路径的索引集合
        set2 = set(all_pidxs)  # 获取所有正样本索引的集合
        neg_smapling_set = list(set1 - set2)  # 获取负样本索引集合

        nidxs = np.random.choice(
            neg_smapling_set, self.cached_queries * 4, replace=False
        )  # 随机选择负样本索引

        for nidx in nidxs:
            neg_samples.append(self.all_file_loc[nidx])  # 添加负样本文件路径
        np.set_printoptions(threshold=np.inf)  # 设置打印选项

        # 创建查询、正样本和负样本的数据加载器
        batch_size = 128
        q_dset = SejongSouthbayLoader(all_file_loc=queries)
        q_loader = DataLoader(
            q_dset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=16,
        )

        p_dset = SejongSouthbayLoader(all_file_loc=pos_samples)
        p_loader = DataLoader(
            p_dset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=16,
        )

        n_dset = SejongSouthbayLoader(all_file_loc=neg_samples)
        n_loader = DataLoader(
            n_dset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=16,
        )

        device = "cuda"  # 设置设备为 CUDA
        net = net.to(device)  # 将网络模型移动到 CUDA 设备
        net.eval()  # 设置网络模型为评估模式
        with torch.inference_mode():  # 禁用梯度计算

            # 初始化描述符
            qvecs = np.zeros((len(q_loader.dataset), outputdim), dtype=np.float32)
            pvecs = np.zeros((len(p_loader.dataset), outputdim), dtype=np.float32)
            nvecs = np.zeros((len(n_loader.dataset), outputdim), dtype=np.float32)

            # 计算描述符并挖掘难负样本
            print("Mining hard negatives")
            count = 0
            for i, batch_data in tqdm(enumerate(n_loader), total=len(n_loader)):
                coord, xyz, feat, batch_number = batch_data
                coord, xyz, feat, batch_number = (
                    coord.to(device),
                    xyz.to(device),
                    feat.to(device),
                    batch_number.to(device),
                )
                local_features, global_descriptor = net(coord, xyz, feat, batch_number)
                nvecs[
                    count : count + batch_number[-1] + 1, :
                ] = global_descriptor.cpu().numpy()
                count += batch_number[-1] + 1
                del (
                    coord,
                    xyz,
                    feat,
                    batch_number,
                    batch_data,
                    local_features,
                    global_descriptor,
                )
                gc.collect()
                torch.cuda.empty_cache()

            count = 0
            for i, batch_data in tqdm(enumerate(q_loader), total=len(q_loader)):
                coord, xyz, feat, batch_number = batch_data
                coord, xyz, feat, batch_number = (
                    coord.to(device),
                    xyz.to(device),
                    feat.to(device),
                    batch_number.to(device),
                )
                local_features, global_descriptor = net(coord, xyz, feat, batch_number)
                qvecs[
                    count : count + batch_number[-1] + 1, :
                ] = global_descriptor.cpu().numpy()
                count += batch_number[-1] + 1
                del (
                    coord,
                    xyz,
                    feat,
                    batch_number,
                    batch_data,
                    local_features,
                    global_descriptor,
                )
                gc.collect()
                torch.cuda.empty_cache()

            count = 0
            for i, batch_data in tqdm(enumerate(p_loader), total=len(p_loader)):
                coord, xyz, feat, batch_number = batch_data
                coord, xyz, feat, batch_number = (
                    coord.to(device),
                    xyz.to(device),
                    feat.to(device),
                    batch_number.to(device),
                )
                local_features, global_descriptor = net(coord, xyz, feat, batch_number)
                pvecs[
                    count : count + batch_number[-1] + 1, :
                ] = global_descriptor.cpu().numpy()
                count += batch_number[-1] + 1
                del (
                    coord,
                    xyz,
                    feat,
                    batch_number,
                    batch_data,
                    local_features,
                    global_descriptor,
                )
                gc.collect()
                torch.cuda.empty_cache()

        faiss_index = faiss.IndexFlatL2(outputdim)  # 创建 FAISS 索引
        faiss_index.add(nvecs)  # 添加负样本描述符到索引
        dNeg_arr, n_ind_arr = faiss_index.search(
            qvecs, self.nNeg + 1
        )  # 搜索最接近查询的 nNeg+1 个负样本
        # dNeg_arr - 距离矩阵
        # n_ind_arr - 对应的索引
        for q in range(len(qidxs)):
            qidx = qidxs[q]
            # 查找该查询的正样本索引（缓存索引域）
            cached_pidx = np.where(np.in1d(pidxs, self.pos_pairs_ind[qidx]))
            # cached_pidx: pidxs 中对应于 self.pos_pairs_ind[qidx] 的索引
            faiss_index = faiss.IndexFlatL2(outputdim)
            faiss_index.add(pvecs[cached_pidx])
            dPos, p_ind = faiss_index.search(qvecs[q : q + 1], 1)
            pidx = pidxs[list(cached_pidx[0])[p_ind.item()]]
            loss = dPos.reshape(-1) - dNeg_arr[q, :].reshape(-1) + self.margin
            violatingNeg = loss > 0
            if self.prev_epoch_neg[qidx][0] == -1:
                # 如果违反的负样本少于 nNeg，则跳过该查询
                if np.sum(violatingNeg) <= self.nNeg:
                    continue
                else:
                    # 选择最难的负样本并更新 prev_epoch_neg
                    hardest_negIdx = np.argsort(loss)[: self.nNeg]
                    # 选择最难的负样本
                    hardestNeg = nidxs[n_ind_arr[q, hardest_negIdx]]
            else:
                # 至少 n/2 个新的负样本
                if np.sum(violatingNeg) <= math.ceil(self.nNeg / 2):
                    continue
                else:
                    # 从随机图像池和上一轮中选择最难的负样本
                    hardest_negIdx = np.argsort(loss)[
                        : min(self.nNeg, np.sum(violatingNeg))
                    ]
                    cached_hardestNeg = nidxs[n_ind_arr[q, hardest_negIdx]]
                    neg_candidates = np.asarray(
                        [
                            x
                            for x in cached_hardestNeg
                            if x not in self.prev_epoch_neg[qidx]
                        ]
                        + self.prev_epoch_neg[qidx]
                    )
                    hardestNeg = neg_candidates[
                        random.sample(range(len(neg_candidates)), self.nNeg)
                    ]
            self.prev_epoch_neg[qidx] = np.copy(hardestNeg).tolist()

            # 转换回原始索引（回到原始索引域）
            q_loc = self.all_file_loc[qidx]
            p_loc = self.all_file_loc[pidx]
            n_loc = self.all_file_loc[hardestNeg[0]]

            # 打包三元组和目标
            triplet_id = [qidx, pidx, hardestNeg[0]]
            triplet = [q_loc, p_loc, n_loc]
            target = [-1, 1, 0]
            self.triplets.append((triplet, triplet_id, target))

    def __len__(self):  # 数据集长度方法
        return len(self.triplets)  # 返回三元组的数量

    def base_2_lidar(self, wTb):  # 基础到激光雷达转换方法
        bTl = np.asarray(
            [
                -0.999982947984152,
                -0.005839838492430,
                -0.000005225706031,
                1.7042,
                0.005839838483221,
                -0.999982947996283,
                0.000001775876813,
                -0.0210,
                -0.000005235987756,
                0.000001745329252,
                0.999999999984769,
                1.8047,
                0,
                0,
                0,
                1,
            ]
        ).reshape(
            4, 4
        )  # 定义基础到激光雷达的转换矩阵
        return wTb @ bTl  # 返回转换后的矩阵

    def get_delta_pose(self, transforms, filename):  # 获取位姿变化方法
        if filename[22] == "A":  # 如果文件名的第 23 个字符是 'A'
            w_T_p1 = transforms[0]  # 获取第一个变换矩阵
            w_T_p2 = transforms[1]  # 获取第二个变换矩阵
        else:
            w_T_p1 = self.base_2_lidar(transforms[0])  # 将第一个变换矩阵转换为激光雷达坐标系
            w_T_p2 = self.base_2_lidar(transforms[1])  # 将第二个变换矩阵转换为激光雷达坐标系

        p1_T_w = np.linalg.inv(w_T_p1)  # 计算第一个变换矩阵的逆矩阵
        p1_T_p2 = np.matmul(p1_T_w, w_T_p2)  # 计算两个变换矩阵之间的变化
        return p1_T_p2  # 返回位姿变化矩阵

    def get_point_tuples(self, q_xyz, p_xyz, q_idx, p_idx, filename):  # 获取点云配对方法
        q_pcd = make_open3d_point_cloud(q_xyz, color=None)  # 创建查询点云
        p_pcd = make_open3d_point_cloud(p_xyz, color=None)  # 创建正样本点云

        matching_search_voxel_size = min(self.voxel_size * 1.5, 0.1)  # 设置匹配搜索体素大小

        q_odom = self.transforms[q_idx]  # 获取查询的变换矩阵
        p_odom = self.transforms[p_idx]  # 获取正样本的变换矩阵
        all_odometry = [q_odom, p_odom]  # 创建变换矩阵列表

        delta_T = self.get_delta_pose(all_odometry, filename)  # 获取位姿变化矩阵
        p_pcd.transform(delta_T)  # 对正样本点云进行变换

        reg = o3d.pipelines.registration.registration_icp(  # 使用 ICP 算法进行点云配准
            p_pcd,
            q_pcd,
            0.2,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
        )
        p_pcd.transform(reg.transformation)  # 对正样本点云进行变换

        pos_pairs = get_matching_indices(  # 获取匹配的点对
            q_pcd, p_pcd, matching_search_voxel_size
        )
        # assert pos_pairs.ndim == 2, f"No pos_pairs for {query_id} in drive id: {drive_id}"

        return pos_pairs  # 返回匹配的点对

    def __getitem__(self, idx):  # 获取数据方法
        anchor_filename, pos_filename, neg_filename = self.triplets[idx][
            0
        ]  # 获取三元组的文件路径
        anchor_idx, pos_idx, neg_idx = self.triplets[idx][1]  # 获取三元组的索引
        labels = self.triplets[idx][1]  # 获取三元组的标签

        anchor_xyzr = self.read_pcd_file(anchor_filename)  # 读取锚点点云数据
        anchor_coords, anchor_xyz, anchor_feats = self.data_prepare(
            anchor_xyzr,
            voxel_size=np.array([self.voxel_size, self.voxel_size, self.voxel_size]),
        )  # 准备锚点数据

        pos_xyzr = self.read_pcd_file(pos_filename)  # 读取正样本点云数据
        pos_coords, pos_xyz, pos_feats = self.data_prepare(
            pos_xyzr,
            voxel_size=np.array([self.voxel_size, self.voxel_size, self.voxel_size]),
        )  # 准备正样本数据

        point_pos_pairs = self.get_point_tuples(
            anchor_xyz, pos_xyz, anchor_idx, pos_idx, anchor_filename
        )  # 获取匹配的点对

        neg_xyzr = self.read_pcd_file(neg_filename)  # 读取负样本点云数据
        neg_coords, neg_xyz, neg_feats = self.data_prepare(
            neg_xyzr,
            voxel_size=np.array([self.voxel_size, self.voxel_size, self.voxel_size]),
        )  # 准备负样本数据

        return (
            anchor_coords,
            anchor_xyz,
            anchor_feats,
            pos_coords,
            pos_xyz,
            pos_feats,
            neg_coords,
            neg_xyz,
            neg_feats,
            labels,
            point_pos_pairs,
        )  # 返回锚点、正样本和负样本的数据及标签和匹配的点对
