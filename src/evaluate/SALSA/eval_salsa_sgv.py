# This script is adapted from: https://github.com/jac99/Egonn/blob/main/eval/evaluate.py

import argparse
import copy
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle
import random
import sys
from itertools import repeat
from time import time
from typing import List, Tuple, Union

import numpy as np
import open3d as o3d
import torch
import tqdm
from sgv_utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data.datasets.southbay.pypcd as pypcd
from data.dataset_utils import voxelize
from models.pca_model import CombinedModel, PCAModel
from models.salsa import SALSA
from data.datasets.base_datasets import (
    EvaluationSet,
    EvaluationTuple,
    get_pointcloud_loader,
)
from data.datasets.kitti360.utils import kitti360_relative_pose
from data.datasets.kitti.utils import get_relative_pose as kitti_relative_pose
from data.datasets.mulran.utils import relative_pose as mulran_relative_pose
from data.datasets.point_clouds_utils import (
    icp,
    make_open3d_feature,
    make_open3d_point_cloud,
    preprocess_pointcloud,
)
from data.datasets.poses_utils import apply_transform, relative_pose

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

print('\n' + ' '.join([sys.executable] + sys.argv))


class Evaluator:
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str,
                 radius: List[float] = (5, 20), k: int = 50, n_samples: int =None, debug: bool = False):
        # radius: list of thresholds (in meters) to consider an element from the map sequence a true positive
        # k: maximum number of nearest neighbours to consider
        # n_samples: number of samples taken from a query sequence (None=all query elements)

        assert os.path.exists(dataset_root), f"Cannot access dataset root: {dataset_root}"
        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        self.eval_set_filepath = os.path.join(os.path.dirname(__file__), '../../data/datasets/',self.dataset_type, eval_set_pickle)
        self.device = device
        self.radius = radius
        self.k = k
        self.n_samples = n_samples
        self.debug = debug

        assert os.path.exists(self.eval_set_filepath), f'Cannot access evaluation set pickle: {self.eval_set_filepath}'
        self.eval_set = EvaluationSet()
        self.eval_set.load(self.eval_set_filepath)
        if debug:
            # Make the same map set and query set in debug mdoe
            self.eval_set.map_set = self.eval_set.map_set[:4]
            self.eval_set.query_set = self.eval_set.map_set[:4]

        if n_samples is None or len(self.eval_set.query_set) <= n_samples:
            self.n_samples = len(self.eval_set.query_set)
        else:
            self.n_samples = n_samples

        self.pc_loader = get_pointcloud_loader(self.dataset_type)

    def evaluate(self, model, *args, **kwargs):
        map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)
        query_embeddings = self.compute_embeddings(self.eval_set.query_set, model)

        map_positions = self.eval_set.get_map_positions()
        query_positions = self.eval_set.get_query_positions()

        # Dictionary to store the number of true positives for different radius and NN number
        tp = {r: [0] * self.k for r in self.radius}
        query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        # Randomly sample n_samples clouds from the query sequence and NN search in the target sequence
        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # Euclidean distance between the query and nn
            delta = query_pos - map_positions[nn_ndx]  # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)  # (k,) array
            # Count true positives for different radius and NN number
            tp = {r: [tp[r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in
                  self.radius}

        recall = {r: [tp[r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        # percentage of 'positive' queries (with at least one match in the map sequence within given radius)
        return {'recall': recall}

    def compute_embedding(self, pc, model, *args, **kwargs):
        # This method must be implemented in inheriting classes
        # Must return embedding as a numpy vector
        raise NotImplementedError('Not implemented')

    def model2eval(self, model):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        model.eval()

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        self.model2eval(model)

        embeddings = None
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
            assert os.path.exists(scan_filepath)
            pc =self.read_pcd_file(scan_filepath)
            # pc = self.pc_loader(scan_filepath)
            pc = torch.tensor(pc)

            embedding = self.compute_embedding(pc, model)
            if embeddings is None:
                embeddings = np.zeros((len(eval_subset), embedding.shape[1]), dtype=embedding.dtype)
            embeddings[ndx] = embedding

        return embeddings


def euclidean_distance(query, database):
    return torch.cdist(torch.tensor(query).unsqueeze(0).unsqueeze(0), torch.tensor(database).unsqueeze(0)).squeeze().numpy()

class MetLocEvaluator(Evaluator):
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str,
                 radius: List[float], k: int = 20, n_samples=None, repeat_dist_th: float = 0.5, voxel_size: float = 0.1,
                 icp_refine: bool = True, debug: bool = False):
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, radius, k, n_samples, debug=debug)
        self.repeat_dist_th = repeat_dist_th
        self.icp_refine = icp_refine
        self.voxel_size = voxel_size

    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        [model.eval() for model in models]

    def evaluate(self, model, d_thresh, *args, **kwargs):
        if 'only_global' in kwargs:
            self.only_global = kwargs['only_global']
        else:
            self.only_global = False

        query_embeddings, local_query_embeddings = self.compute_embeddings(self.eval_set.query_set, model) # same for Nquery

        map_embeddings, local_map_embeddings = self.compute_embeddings(self.eval_set.map_set, model) # Nmap x 256 , Nmap x Dict{'keypoints':torch128x3, 'features':torch128x128}

        map_positions = self.eval_set.get_map_positions() # Nmap x 2
        query_positions = self.eval_set.get_query_positions() # Nquery x 2

        if self.n_samples is None or len(query_embeddings) <= self.n_samples:
            query_indexes = list(range(len(query_embeddings)))
            self.n_samples = len(query_embeddings)
        else:
            query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        if self.only_global:
            metrics = {}
        else:
            metrics = {eval_mode: {'rre': [], 'rte': [], 'repeatability': [],
                                'success': [], 'success_inliers': [], 'failure_inliers': [],
                                'rre_refined': [], 'rte_refined': [], 'success_refined': [],
                                'success_inliers_refined': [], 'repeatability_refined': [],
                                'failure_inliers_refined': [], 't_ransac': []}
                       for eval_mode in ['Initial', 'Re-Ranked']}

        # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
        global_metrics = {'tp': {r: [0] * self.k for r in self.radius}}
        global_metrics['tp_rr'] = {r: [0] * self.k for r in self.radius}
        global_metrics['RR'] = {r: [] for r in self.radius}
        global_metrics['RR_rr'] = {r: [] for r in self.radius}
        global_metrics['t_RR'] = []

        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]
            query_pose = self.eval_set.query_set[query_ndx].pose

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)

            nn_ndx = np.argsort(embed_dist)[:self.k]

            # PLACE RECOGNITION EVALUATION
            # Euclidean distance between the query and nn
            # Here we use non-icp refined poses, but for the global descriptor it's fine
            delta = query_pos - map_positions[nn_ndx]       # (k, 2) array


            euclid_dist = np.linalg.norm(delta, axis=1)     # (k,) array

            # re_rank = True
            if d_thresh > 0:
                # fitness_list = []
                # fitness_list = np.zeros(len(nn_ndx))
                topk = min(self.k,len(nn_ndx))
                fitness_list = np.zeros(topk)
                tick = time()
                for k in range(topk):
                    k_id = nn_ndx[k]
                    conf_val = sgv_fn(local_query_embeddings[query_ndx], local_map_embeddings[k_id], d_thresh=d_thresh)
                    fitness_list[k] = conf_val
                topk_rerank = np.flip(np.asarray(fitness_list).argsort())
                topk_rerank_inds = copy.deepcopy(nn_ndx)
                topk_rerank_inds[:topk] = nn_ndx[topk_rerank]
                t_rerank = time() - tick
                global_metrics['t_RR'].append(t_rerank)

                delta_rerank = query_pos - map_positions[topk_rerank_inds]
                euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)
            else:
                euclid_dist_rr = euclid_dist
                global_metrics['t_RR'].append(0)

            # Count true positives for different radius and NN number
            global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            global_metrics['tp_rr'] = {r: [global_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            global_metrics['RR'] = {r: global_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist <= r) if x), 0)] for r in self.radius}
            global_metrics['RR_rr'] = {r: global_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in self.radius}
            if self.only_global:
                continue
            # continue
            # METRIC LOCALIZATION EVALUATION

            for eval_mode in ['Initial', 'Re-Ranked']:
                # Get the first match and compute local stats only for the best match
                if eval_mode == 'Initial':
                    nn_idx = nn_ndx[0]
                elif eval_mode == 'Re-Ranked':
                    nn_idx = topk_rerank_inds[0]

                # Ransac alignment
                tick = time()
                T_estimated, inliers, _ = self.ransac_fn(local_query_embeddings[query_ndx],
                                                         local_map_embeddings[nn_idx])

                t_ransac = time() - tick

                nn_pose = self.eval_set.map_set[nn_idx].pose
                # T_gt is np.linalg.inv(nn_pose) @ query_pose
                if self.dataset_type == 'mulran':
                    T_gt = mulran_relative_pose(query_pose, nn_pose)
                elif self.dataset_type == 'southbay':
                    T_gt = relative_pose(query_pose, nn_pose)
                elif self.dataset_type == 'kitti':
                    T_gt = kitti_relative_pose(query_pose, nn_pose)
                elif self.dataset_type == 'alita':
                    T_gt = relative_pose(query_pose, nn_pose)
                elif self.dataset_type == 'kitti360':
                    T_gt = kitti360_relative_pose(query_pose, nn_pose)
                else:
                    raise NotImplementedError('Unknown dataset type')
                
                # ? DEBUG
                # for n in nn_ndx:
                #     cur_pose = self.eval_set.map_set[n].pose
                #     T_ = mulran_relative_pose(query_pose, cur_pose)
                #     T_gtt = mulran_relative_pose(query_pose, nn_pose)
                #     print(n)
                #     print(nn_idx)
                #     delta_pose = cur_pose - nn_pose
                #     print(np.linalg.norm(delta_pose))
                #     print(np.linalg.norm(delta_pose[:3, 3]))
                #     print(np.linalg.norm(T_ - T_gtt))
                #     print(np.linalg.norm(T_ - T_gt))
                # second_pose = self.eval_set.map_set[nn_ndx[1]].pose

                # Refine the pose using ICP
                if not self.icp_refine:
                    T_refined = T_gt
                else:
                    query_filepath = os.path.join(self.dataset_root, self.eval_set.query_set[query_ndx].rel_scan_filepath)
                    query_pc = self.pc_loader(query_filepath)
                    map_filepath = os.path.join(self.dataset_root, self.eval_set.map_set[nn_idx].rel_scan_filepath)
                    map_pc = self.pc_loader(map_filepath)
                    if self.dataset_type in ['mulran', 'kitti', 'kitti360']:
                        query_pc = preprocess_pointcloud(query_pc, remove_zero_points=True,
                                                         min_x=-80, max_x=80, min_y=-80, max_y=80, min_z=-0.9)
                        map_pc = preprocess_pointcloud(map_pc, remove_zero_points=True,
                                                         min_x=-80, max_x=80, min_y=-80, max_y=80, min_z=-0.9)
                    elif self.dataset_type in ['southbay', 'alita']:
                        # -1.6 removes most of the ground plane
                        query_pc = preprocess_pointcloud(query_pc, remove_zero_points=True,
                                                         min_x=-100, max_x=100, min_y=-100, max_y=100, min_z=-1.6)
                        map_pc = preprocess_pointcloud(map_pc, remove_zero_points=True,
                                                       min_x=-100, max_x=100, min_y=-100, max_y=100, min_z=-1.6)
                    else:
                        raise NotImplementedError(f"Unknown dataset type: {self.dataset_type}")
                    T_refined, _, _ = icp(query_pc, map_pc, T_gt)

                # Compute repeatability using refined pose
                kp1 = local_query_embeddings[query_ndx]['keypoints']
                kp2 = local_map_embeddings[nn_idx]['keypoints']
                metrics[eval_mode]['repeatability'].append(calculate_repeatability(kp1, kp2, T_gt, threshold=self.repeat_dist_th))
                metrics[eval_mode]['repeatability_refined'].append(calculate_repeatability(kp1, kp2, T_refined, threshold=self.repeat_dist_th))

                # calc errors
                rte = np.linalg.norm(T_estimated[:3, 3] - T_gt[:3, 3])


                # ? DEBUG
                # 输出距离查询点最近点的真实坐标
                nearest_point_xyz = nn_pose[:3, 3]
                nearest_point_xy = nn_pose[:2, 3]
                print(f"Nearest point XYZ: {nearest_point_xyz}")
                print(f"Nearest point XY: {nearest_point_xy}")

                # 输出查询点的真实坐标
                query_point_xyz = query_pose[:3, 3]
                query_point_xy = query_pose[:2, 3]
                print(f"Query point XYZ: {query_point_xyz}")
                print(f"Query point XY: {query_point_xy}")

                # 输出二者之间的距离
                xyz_dis = np.linalg.norm(nearest_point_xyz - query_point_xyz)
                xy_dis = np.linalg.norm(nearest_point_xy - query_point_xy)
                print(xyz_dis)
                print(xy_dis)
                # ----------------------------------------------------------------------------------------------
                # 将矩阵作用于点
                # 这两种写法的结果是一样的
                # _ = query_pose[:3, 3] = query_pose[:3, 3] @ T_estimated[:3, :3].transpose(1, 0) + T_estimated[:3, -1]
                # transformed_pose = T_estimated @ query_pose_homogeneous
                # 好像不用作用？
                Ts = {'T_estimated': T_estimated, 'T_gt': T_gt}
                poses = {'query_pose': query_pose, 'nn_pose': nn_pose}

                for T_name, T in Ts.items():
                    for pose_name, pose in poses.items():
                        transformed_pose_1 = T @ pose
                        transformed_pose_2 = pose @ T
                        # print(f"{T} @ \n{pose} = \n{transformed_pose_1}")
                        # print(f"{pose} @ \n{T} = \n{transformed_pose_2}")
                        distance_xyz_1 = np.linalg.norm(transformed_pose_1[:3, 3] - nn_pose[:3, 3])
                        distance_xy_1 = np.linalg.norm(transformed_pose_1[:2, 3] - nn_pose[:2, 3])
                        distance_xyz_2 = np.linalg.norm(transformed_pose_2[:3, 3] - nn_pose[:3, 3])
                        distance_xy_2 = np.linalg.norm(transformed_pose_2[:2, 3] - nn_pose[:2, 3])
                        print(f"Distance in XYZ (using {T_name} @ {pose_name}): {distance_xyz_1}")
                        print(f"Distance in XY (using {T_name} @ {pose_name}): {distance_xy_1}")
                        print(f"Distance in XYZ (using {pose_name} @ {T_name}): {distance_xyz_2}")
                        print(f"Distance in XY (using {pose_name} @ {T_name}): {distance_xy_2}")
                        """
                        Distance in XYZ (using T_estimated @ query_pose): 287318.49081961164
                        Distance in XY (using T_estimated @ query_pose): 287010.11762801063
                        Distance in XYZ (using query_pose @ T_estimated): 143683.2229467801
                        Distance in XY (using query_pose @ T_estimated): 143545.60558230107
                        Distance in XYZ (using T_estimated @ nn_pose): 143683.861823666
                        Distance in XY (using T_estimated @ nn_pose): 143546.24681926807

                        Distance in XYZ (using nn_pose @ T_estimated): 4.653481912855474
                        Distance in XY (using nn_pose @ T_estimated): 4.653385523107071

                        Distance in XYZ (using T_gt @ query_pose): 278815.467154478
                        Distance in XY (using T_gt @ query_pose): 277618.77041948296
                        Distance in XYZ (using query_pose @ T_gt): 143683.86870899878
                        Distance in XY (using query_pose @ T_gt): 143546.2549886668
                        Distance in XYZ (using T_gt @ nn_pose): 135484.80388246189
                        Distance in XY (using T_gt @ nn_pose): 134136.63551571083

                        Distance in XYZ (using nn_pose @ T_gt): 4.794887719954771
                        Distance in XY (using nn_pose @ T_gt): 4.794753240919318
                        """
                # ----------------------------------------------------------------------------------------------


                cos_rre = (np.trace(T_estimated[:3, :3].transpose(1, 0) @ T_gt[:3, :3]) - 1.) / 2.
                rre = np.arccos(np.clip(cos_rre, a_min=-1., a_max=1.)) * 180. / np.pi

                metrics[eval_mode]['t_ransac'].append(t_ransac)    # RANSAC time

                # 2 meters and 5 degrees threshold for successful registration
                if rte > 2.0 or rre > 5.:
                    metrics[eval_mode]['success'].append(0.)
                    metrics[eval_mode]['failure_inliers'].append(inliers)
                else:
                    metrics[eval_mode]['success'].append(1.)
                    metrics[eval_mode]['success_inliers'].append(inliers)
                    metrics[eval_mode]['rte'].append(rte)
                    metrics[eval_mode]['rre'].append(rre)

                if self.icp_refine:
                    # calc errors using refined pose
                    rte_refined = np.linalg.norm(T_estimated[:3, 3] - T_refined[:3, 3])
                    cos_rre_refined = (np.trace(T_estimated[:3, :3].transpose(1, 0) @ T_refined[:3, :3]) - 1.) / 2.
                    rre_refined = np.arccos(np.clip(cos_rre_refined, a_min=-1., a_max=1.)) * 180. / np.pi

                    # 2 meters and 5 degrees threshold for successful registration
                    if rte_refined > 2.0 or rre_refined > 5.:
                        metrics[eval_mode]['success_refined'].append(0.)
                        metrics[eval_mode]['failure_inliers_refined'].append(inliers)
                    else:
                        metrics[eval_mode]['success_refined'].append(1.)
                        metrics[eval_mode]['success_inliers_refined'].append(inliers)
                        metrics[eval_mode]['rte_refined'].append(rte_refined)
                        metrics[eval_mode]['rre_refined'].append(rre_refined)

        # Calculate mean metrics
        global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        global_metrics["recall_rr"] = {r: [global_metrics['tp_rr'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        global_metrics['MRR'] = {r: np.mean(np.asarray(global_metrics['RR'][r])) for r in self.radius}
        global_metrics['MRR_rr'] = {r: np.mean(np.asarray(global_metrics['RR_rr'][r])) for r in self.radius}
        global_metrics['mean_t_RR'] = np.mean(np.asarray(global_metrics['t_RR']))

        mean_metrics = {}
        if not self.only_global:
            # Calculate mean values of local descriptor metrics
            for eval_mode in ['Initial', 'Re-Ranked']:
                mean_metrics[eval_mode] = {}
                for metric in metrics[eval_mode]:
                    m_l = metrics[eval_mode][metric]
                    if len(m_l) == 0:
                        mean_metrics[eval_mode][metric] = 0.
                    else:
                        if metric == 't_ransac':
                            mean_metrics[eval_mode]["t_ransac_sd"] = np.std(m_l)
                        mean_metrics[eval_mode][metric] = np.mean(m_l)

        return global_metrics, mean_metrics

    def ransac_fn(self, query_keypoints, candidate_keypoints):
        """

        Returns fitness score and estimated transforms
        Estimation using Open3d 6dof ransac based on feature matching.
        """
        kp1 = query_keypoints['keypoints']
        kp2 = candidate_keypoints['keypoints']
        ransac_result = get_ransac_result(query_keypoints['features'], candidate_keypoints['features'],
                                          kp1, kp2)
        return ransac_result.transformation, len(ransac_result.correspondence_set), ransac_result.fitness

    def data_prepare(self, xyzr, voxel_size=np.array([0.1, 0.1, 0.1])):

        lidar_pc = copy.deepcopy(xyzr)
        coords = np.round(lidar_pc[:, :3] / voxel_size)
        coords_min = coords.min(0, keepdims=1)
        coords -= coords_min
        feats = lidar_pc

        hash_vals, _, uniq_idx = sparse_quantize(coords, return_index=True, return_hash=True)
        coord_voxel, feat = coords[uniq_idx], feats[uniq_idx]
        coord = copy.deepcopy(feat[:,:3])

        coord = torch.FloatTensor(coord)
        feat = torch.FloatTensor(feat)
        coord_voxel = torch.LongTensor(coord_voxel)
        return coord_voxel, coord, feat

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model):
        model.eval()
        global_embeddings = None
        local_embeddings = []
        for ndx, e in tqdm.tqdm(enumerate(eval_subset),total = len(eval_subset)):
            scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)

            ################ SALSA ###################################################################
            pc = self.pc_loader(scan_filepath)

            coords, xyz, feats = self.data_prepare(pc,voxel_size = np.array([self.voxel_size,self.voxel_size,self.voxel_size]))
            points = copy.deepcopy(feats)

            batch_number = torch.ones(feats.shape[0]).to(torch.int)
            coords, xyz, feats, points, batch_number = coords.cuda(), xyz.cuda(), feats.cuda(), points.cuda(), batch_number.cuda()

            with torch.inference_mode():
                global_embedding, keypoints, key_embeddings = self.compute_embedding(model,[coords, xyz, feats, batch_number], points, is_dense=True)
            ###########################################################################################

            if global_embeddings is None:
                global_embeddings = np.zeros((len(eval_subset), global_embedding.shape[1]), dtype=global_embedding.dtype)

            global_embeddings[ndx] = global_embedding
            local_embeddings.append({'keypoints': keypoints, 'features': key_embeddings})

        return global_embeddings, local_embeddings

    def compute_embedding(self, model, batch_data, points, is_dense=False):
        """
        Returns global embedding (np.array) as well as keypoints and corresponding descriptors (torch.tensors)
        """
        output_feats, output_desc = model(batch_data)
        output_feats = output_feats[0]

        output_features = output_feats.cpu().detach().numpy()
        output_points = points

        global_descriptor = output_desc.cpu().detach().numpy()
        global_embedding = np.reshape(global_descriptor, (1, -1))

        return global_embedding, output_points.clone().detach().cpu(), torch.tensor(output_features, dtype=torch.float).cpu()

    def print_results(self, global_metrics, metrics):
        # Global descriptor results are saved with the last n_k entry
        print('\n','Initial Retrieval:')
        recall = global_metrics['recall']
        for r in recall:
            print(f"Radius: {r} [m] : ")
            print(f"Recall@N : ", end='')
            for x in recall[r]:
                print("{:0.1f}, ".format(x*100.0), end='')
            print("")
            print('MRR: {:0.1f}'.format(global_metrics['MRR'][r]*100.0))

        print('\n','Re-Ranking:')
        recall_rr = global_metrics['recall_rr']
        for r_rr in recall_rr:
            print(f"Radius: {r_rr} [m] : ")
            print(f"Recall@N : ", end='')
            for x in recall_rr[r_rr]:
                print("{:0.1f}, ".format(x*100.0), end='')
            print("")
            print('MRR: {:0.1f}'.format(global_metrics['MRR_rr'][r_rr]*100.0))
        print('Re-Ranking Time: {:0.3f}'.format(1000.0 *global_metrics['mean_t_RR']))

        print('\n','Metric Localization:')
        for eval_mode in ['Initial', 'Re-Ranked']:
            if eval_mode not in metrics:
                break
            print('#keypoints: {}'.format(eval_mode))
            for s in metrics[eval_mode]:
                print(f"{s}: {metrics[eval_mode][s]:0.3f}")
            print('')


def get_ransac_result(feat1, feat2, kp1, kp2, ransac_dist_th=0.5, ransac_max_it=10000):
    feature_dim = feat1.shape[1]
    pcd_feat1 = make_open3d_feature(feat1, feature_dim, feat1.shape[0])
    pcd_feat2 = make_open3d_feature(feat2, feature_dim, feat2.shape[0])
    if not isinstance(kp1, np.ndarray):
        pcd_coord1 = make_open3d_point_cloud(kp1.numpy())
        pcd_coord2 = make_open3d_point_cloud(kp2.numpy())
    else:
        pcd_coord1 = make_open3d_point_cloud(kp1)
        pcd_coord2 = make_open3d_point_cloud(kp2)

    # ransac based eval
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_coord1, pcd_coord2, pcd_feat1, pcd_feat2,
        mutual_filter=True,
        max_correspondence_distance=ransac_dist_th,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist_th)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_max_it, 0.999))

    return ransac_result


def calculate_repeatability(kp1, kp2, T_gt, threshold: float):
    # Transform the source point cloud to the same position as the target cloud
    kp1_pos_trans = apply_transform(kp1, torch.tensor(T_gt, dtype=torch.float))
    dist = torch.cdist(kp1_pos_trans, kp2)      # (n_keypoints1, n_keypoints2) tensor

    # *** COMPUTE REPEATABILITY ***
    # Match keypoints from the first cloud with closests keypoints in the second cloud
    min_dist, _ = torch.min(dist, dim=1)
    # Repeatability with a distance threshold th
    return torch.mean((min_dist <= threshold).float()).item()

def sparse_quantize(coords,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False,
                    return_hash: bool = False) -> List[np.ndarray]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    hash_vals, indices, inverse_indices = np.unique(ravel_hash(coords),
                                            return_index=True,
                                            return_inverse=True)
    coords = coords[indices]

    if return_hash: outputs = [hash_vals, coords]
    else: outputs = [coords]

    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs

def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x -= np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')
    del model_parameters, params

def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MinkLoc model')
    parser.add_argument('--salsa_model', type=str, required=False, default='model_26.pth')
    parser.add_argument('--dataset_root', type=str, required=False, default='/home/ljc/Dataset/KITTI', help='Path to the dataset root')
    parser.add_argument('--dataset_type', type=str, required=False, default='kitti', choices=['mulran', 'southbay', 'kitti', 'alita', 'kitti360'])
    parser.add_argument('--mulran_sequence', type=str, required=False, default='sejong', choices=['sejong','dcc'])
    parser.add_argument('--eval_set', type=str, required=False, default='kitti_00_eval.pickle', help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--radius', type=float, nargs='+', default=[5, 20], help='True Positive thresholds in meters')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
    parser.add_argument('--weights', type=str, default='/logg3d.pth')
    parser.add_argument('--model', type=str, default='logg3d1k', choices=['logg3d', 'logg3d1k'])
    parser.add_argument('--d_thresh', type=float, default=0.4, help='Dist thresholds in meters')
    parser.add_argument('--n_topk', type=int, default=20, help='Dist thresholds in meters')
    parser.add_argument('--icp_refine', dest='icp_refine', action='store_true')
    parser.add_argument('--only_global', type=bool, default=False, help='If False, also evaluates metric localization')
    parser.set_defaults(icp_refine=True)
    parser.add_argument('--voxel_size', type=float, default=0.5)

    args = parser.parse_args()
    # print(args)
    for arg, value in vars(args).items():
        print(f"--{arg} {value}")
    
    if args.dataset_type == 'kitti':
        args.eval_set = 'kitti_00_eval.pickle'
        # args.dataset_root = '/data/raktim/Datasets/KITTI/dataset/'
    elif args.dataset_type == 'mulran':
        if args.mulran_sequence == 'sejong':
            # args.dataset_root = '/data/raktim/Datasets/Mulran/Sejong'
            args.eval_set = 'test_Sejong1_Sejong2_0.2_20.pickle'
        else:
            # args.dataset_root = '/data/raktim/Datasets/Mulran/DCC'
            args.eval_set = 'test_DCC1_DCC2_10.0_5.pickle'
    elif args.dataset_type == 'southbay':
        # args.dataset_root = '/data/raktim/Datasets/Apollo-Southbay'
        args.eval_set = 'test_SunnyvaleBigloop_1.0_5_20m.pickle'
    elif args.dataset_type == 'alita':
        # args.dataset_root = '/data/raktim/Datasets/ALITA/VAL'
        args.eval_set = 'test_val_5_0.01_5.pickle'
    elif args.dataset_type == 'kitti360':
        # args.dataset_root = '/data/raktim/Datasets/KITTI360/KITTI-360/data_3d_raw'
        args.eval_set = 'kitti360_09_3.0_eval.pickle'

    args.weights = os.path.dirname(__file__) + args.weights
    # print(f'Dataset root: {args.dataset_root}')
    # print(f'Dataset type: {args.dataset_type}')
    # print(f'Evaluation set: {args.eval_set}')
    # print(f'Radius: {args.radius} [m]')
    # print(f'd_thresh: {args.d_thresh} [m]')
    # print(f'n_topk: {args.n_topk} ')
    # print('')

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))
    # device = "cpu"

    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


    # ####################### SALSA + PCA #################################################################
    model = CombinedModel(voxel_sz=args.voxel_size,num_in_features=512,num_out_features=256)
    # salsa_model_save_path = os.path.join(os.path.dirname(__file__),'../../checkpoints/SALSA/Model/model_30.pth')
    salsa_model_save_path= os.path.join(os.path.dirname(__file__),'../../checkpoints/SALSA/Model/', args.salsa_model)
    print(f'salsa_model_save_path: {salsa_model_save_path} ')
    checkpoint = torch.load(salsa_model_save_path)  # ,map_location='cuda:0')
    model.spherelpr.load_state_dict(checkpoint)

    salsa_pca_save_path = os.path.join(os.path.dirname(__file__),'../../checkpoints/SALSA/PCA/pca_model_2.pth')
    model.pca_model.load_state_dict(torch.load(salsa_pca_save_path))

    model = model.to(device)
    evaluator = MetLocEvaluator(args.dataset_root, args.dataset_type, args.eval_set, device, radius=args.radius, k=args.n_topk,
                                   n_samples=args.n_samples, voxel_size=args.voxel_size,
                                   icp_refine=args.icp_refine, debug=False)

    start_time = time()                                   

    global_metrics, metrics = evaluator.evaluate(model, d_thresh=args.d_thresh, only_global=args.only_global)

    end_time = time()
    total_time = end_time - start_time
    print(f"Evaluation time: {total_time:.2f} seconds")


    start_time = time()                                   

    evaluator.print_results(global_metrics, metrics)

    end_time = time()
    total_time = end_time - start_time
    print(f"Evaluation time: {total_time:.2f} seconds")

