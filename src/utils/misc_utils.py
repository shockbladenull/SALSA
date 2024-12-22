import csv
import random

import numpy as np
import torch
import yaml


def collate_fn(batch):
    # 将batch中的点云数据（坐标、3D 点、特征）合并为统一张量，并生成每个点所属样本的标记。

    # batch 是一个列表，包含 128 个三元组 (coord, xyz, feat)
    # 以coord为例,coord是一个128的tuple
    # coord[0]是一个(6579,3)的tensor
    # 其中 coord 是点云的坐标，xyz 是点的 3D 空间位置，feat 是点的特征
    coord, xyz, feat = list(zip(*batch))  # 将 batch 中的三元组分别解压为 coord、xyz、feat 三个列表

    offset, count = [], 0  # 初始化偏移量列表 offset 和累积点计数器 count

    new_coord, new_xyz, new_feat = [], [], []  # 用于存储合并前的各样本坐标、3D 点和特征
    k = 0  # 样本计数器
    for i, item in enumerate(xyz):  # 遍历每个样本的 xyz 数据（3D 点位置）

        count += item.shape[0]  # 累加当前样本的点数量
        k += 1  # 样本计数加 1
        offset.append(count)  # 将当前累积点数记录为偏移量
        new_coord.append(coord[i])  # 收集当前样本的坐标
        new_xyz.append(xyz[i])  # 收集当前样本的 3D 点
        new_feat.append(feat[i])  # 收集当前样本的特征

    # 生成 offset_，表示每个样本的点数量
    offset_ = torch.IntTensor(offset[:k]).clone()  # 转换为 PyTorch 张量，表示前 k 个样本的偏移量
    offset_[1:] = offset_[1:] - offset_[:-1]  # 计算每个样本的点数量（相邻偏移量的差值）

    # 生成 batch_number，标记每个点属于哪个样本
    batch_number = torch.cat(
        [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0  # 为每个样本的所有点分配编号 ii
    ).long()  # 将结果拼接成一个张量，并转换为 long 类型

    # 将所有样本的坐标、3D 点和特征合并为一个张量
    coords, xyz, feat = (
        torch.cat(new_coord[:k]),  # 合并前 k 个样本的坐标
        torch.cat(new_xyz[:k]),  # 合并前 k 个样本的 3D 点
        torch.cat(new_feat[:k]),  # 合并前 k 个样本的特征
    )

    # 返回合并后的坐标、3D 点、特征张量，以及点对应的样本编号张量
    # coords.shape = torch.Size([736879, 3])
    return coords, xyz, feat, batch_number


def tuple_collate_fn(batch):
    # 以coord为例,coord是一个12的tuple
    # coord[0]是一个(5759,3)的tensor
    (
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
    ) = list(zip(*batch))

    # 初始化偏移量和计数器
    offset, count = [], 0

    # 初始化合并后的点云坐标、3D 点、特征、标签和点对信息列表
    new_coord, new_xyz, new_feat, new_label, new_point_pos_pairs = [], [], [], [], []

    # 处理 Anchor 部分的点云数据
    coord, xyz, feat = anchor_coords, anchor_xyz, anchor_feats
    for i, item in enumerate(xyz):  # 遍历每个样本中的 Anchor 部分
        count += item.shape[0]  # 累加点的数量
        offset.append(count)  # 添加当前点的累积偏移量
        new_coord.append(coord[i])  # 添加 Anchor 坐标
        new_xyz.append(xyz[i])  # 添加 Anchor 3D 点
        new_feat.append(feat[i])  # 添加 Anchor 特征
        new_label.append(labels[i][0])  # 添加 Anchor 的标签

    # 处理 Positive 部分的点云数据
    coord, xyz, feat = pos_coords, pos_xyz, pos_feats
    for i, item in enumerate(xyz):  # 遍历每个样本中的 Positive 部分
        count += item.shape[0]  # 累加点的数量
        offset.append(count)  # 添加当前点的累积偏移量
        new_coord.append(coord[i])  # 添加 Positive 坐标
        new_xyz.append(xyz[i])  # 添加 Positive 3D 点
        new_feat.append(feat[i])  # 添加 Positive 特征
        new_label.append(labels[i][1])  # 添加 Positive 的标签

    # 处理 Negative 部分的点云数据
    coord, xyz, feat = neg_coords, neg_xyz, neg_feats
    for i, item in enumerate(xyz):  # 遍历每个样本中的 Negative 部分
        count += item.shape[0]  # 累加点的数量
        offset.append(count)  # 添加当前点的累积偏移量
        new_coord.append(coord[i])  # 添加 Negative 坐标
        new_xyz.append(xyz[i])  # 添加 Negative 3D 点
        new_feat.append(feat[i])  # 添加 Negative 特征
        new_label.append(labels[i][2])  # 添加 Negative 的标签

    # 如果点对信息存在，处理 point_pos_pairs
    if point_pos_pairs != None:
        for i, item in enumerate(point_pos_pairs):  # 遍历每个样本的点对信息
            new_point_pos_pairs.append(item)  # 将点对信息添加到合并后的列表中

    # 计算偏移量张量，用于后续生成批次编号
    offset_ = torch.IntTensor(offset).clone()
    offset_[1:] = offset_[1:] - offset_[:-1]  # 计算每个样本中的点数
    # 创建批次编号张量，表示每个点属于哪一个样本
    batch_number = torch.cat(
        [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
    ).long()

    # 将所有点云数据、特征和标签合并为单一张量
    coords, xyz, feat, labels = (
        torch.cat(new_coord),  # 合并所有样本的坐标
        torch.cat(new_xyz),  # 合并所有样本的 3D 点
        torch.cat(new_feat),  # 合并所有样本的特征
        torch.Tensor(new_label),  # 合并所有样本的标签
    )

    # 返回合并后的点云数据和其他辅助信息
    return coords, xyz, feat, batch_number, labels, new_point_pos_pairs


def hashM(arr, M):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M**d
        else:
            hash_vec += arr[d] * M**d
    return hash_vec


def pdist(A, B, dist_type="L2"):
    if dist_type == "L2":
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == "SquareL2":
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError("Not implemented")


#####################################################################################
# Load poses
#####################################################################################


def load_poses_from_csv(file_name):
    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        data_poses = list(reader)

    transforms = []
    positions = []
    for cnt, line in enumerate(data_poses):
        line_f = [float(i) for i in line]
        P = np.vstack((np.reshape(line_f[1:], (3, 4)), [0, 0, 0, 1]))
        transforms.append(P)
        positions.append([P[0, 3], P[1, 3], P[2, 3]])
    return np.asarray(transforms), np.asarray(positions)


#####################################################################################
# Load timestamps
#####################################################################################


def load_timestamps_csv(file_name):
    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        data_poses = list(reader)
    data_poses_ts = np.asarray([float(t) / 1e9 for t in np.asarray(data_poses)[:, 0]])
    return data_poses_ts


def read_yaml_config(filename):
    with open(filename, "r") as stream:
        try:
            # Load the YAML file
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None
