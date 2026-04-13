import os
import glob
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import argparse

import sys
sys.path.insert(0, os.path.dirname(__file__))
from u5_box_utils import fill_box_interior, points_in_boxes

# U5: Ouster OS2, 地面在 z ≈ 0m
GT_BOUNDS = [-40.0, -40.0, -2.0, 40.0, 40.0, 6.0]
GT_VOXEL  = 0.4
GT_GRID   = (200, 200, 20)  # Z 從 -2.0 到 6.0 → 8.0m / 0.4 = 20 layers

# Labels
LBL_FREE = 17
LBL_MANMADE = 15
LBL_ROAD = 11

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def generate_occupancy_raw(dataroot, pose_dict, num_sweeps=40, out_dir='output'):
    """純 360° 原始 LiDAR 堆疊版，動態物件用 3D Box 填充"""
    pcd_files = sorted(glob.glob(os.path.join(dataroot, 'os_2_points', '*.pcd')))
    print(f"[INFO] RAW 模式：360° LiDAR 堆疊 + 3D Box 填充 (無語意)")
    
    box_data = load_pkl(os.path.join(dataroot, '3dbox_result.pkl'))
    box_dict = {}
    for item in box_data:
        frame_id = item['file_name'].replace('.pcd', '')
        valid = item['score'] >= 0.4
        box_dict[frame_id] = {
            'names': item['name'][valid],
            'boxes_lidar': item['boxes_lidar'][valid],
            'pred_labels': item['pred_labels'][valid]
        }
    
    frames = [os.path.splitext(os.path.basename(f))[0] for f in pcd_files]
    total_filled = 0
    os.makedirs(out_dir, exist_ok=True)
    
    for i, frame_id in enumerate(tqdm(frames, desc='Generating RAW Occupancy (U5)')):
        curr_pose = pose_dict[frame_id]['matrix']
        curr_pose_inv = np.linalg.inv(curr_pose)
        curr_boxes = box_dict.get(frame_id, {'names': [], 'boxes_lidar': []})
        
        min_b = np.array(GT_BOUNDS[:3])
        max_b = np.array(GT_BOUNDS[3:])
        
        pcd = o3d.io.read_point_cloud(pcd_files[i])
        curr_pts = np.asarray(pcd.points)
        pcs = [curr_pts]
        
        for j in range(1, num_sweeps + 1):
            if i - j < 0: break
            prev_id = frames[i - j]
            prev_pose = pose_dict[prev_id]['matrix']
            prev_pcd = o3d.io.read_point_cloud(pcd_files[i - j])
            prev_pts = np.asarray(prev_pcd.points)
            prev_boxes = box_dict.get(prev_id, {'boxes_lidar': []})['boxes_lidar']
            if len(prev_boxes) > 0:
                mask = points_in_boxes(prev_pts, prev_boxes)
                prev_pts = prev_pts[~mask]
            if len(prev_pts) > 0:
                p_homo = np.hstack((prev_pts, np.ones((len(prev_pts), 1))))
                transformed = (curr_pose_inv @ prev_pose @ p_homo.T).T[:, :3]
                pcs.append(transformed)
        
        for j in range(1, num_sweeps + 1):
            if i + j >= len(frames): break
            next_id = frames[i + j]
            next_pose = pose_dict[next_id]['matrix']
            next_pcd = o3d.io.read_point_cloud(pcd_files[i + j])
            next_pts = np.asarray(next_pcd.points)
            next_boxes = box_dict.get(next_id, {'boxes_lidar': []})['boxes_lidar']
            if len(next_boxes) > 0:
                mask = points_in_boxes(next_pts, next_boxes)
                next_pts = next_pts[~mask]
            if len(next_pts) > 0:
                p_homo = np.hstack((next_pts, np.ones((len(next_pts), 1))))
                transformed = (curr_pose_inv @ next_pose @ p_homo.T).T[:, :3]
                pcs.append(transformed)
        
        all_pts = np.vstack(pcs)
        
        mask = np.all((all_pts >= min_b) & (all_pts < max_b), axis=1)
        valid_xyz = all_pts[mask]
        idxs = np.clip(((valid_xyz - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
        occ = np.ones(GT_GRID, dtype=np.uint8) * LBL_FREE
        occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = LBL_MANMADE
        
        if len(curr_boxes['names']) > 0:
            occ, filled = fill_box_interior(occ, curr_boxes['boxes_lidar'], curr_boxes['names'], GT_BOUNDS, GT_VOXEL, GT_GRID)
            total_filled += filled
        
        save_path = os.path.join(out_dir, frame_id)
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(os.path.join(save_path, 'labels.npz'), semantics=occ)
    
    print(f"\n[RAW Pipeline Complete] 總共有 {total_filled} 個動態物件 Voxel 被 3D Box 霸道填滿！")

def generate_occupancy_heuristic(dataroot, pose_dict, num_sweeps=40, out_dir='output'):
    """360° LiDAR 堆疊 + 高度啟發式判斷道路 vs 建築"""
    pcd_files = sorted(glob.glob(os.path.join(dataroot, 'os_2_points', '*.pcd')))
    print(f"[INFO] HEURISTIC 模式：360° LiDAR + 高度判斷道路 + 3D Box 填充")
    
    box_data = load_pkl(os.path.join(dataroot, '3dbox_result.pkl'))
    box_dict = {}
    for item in box_data:
        frame_id = item['file_name'].replace('.pcd', '')
        valid = item['score'] >= 0.4
        box_dict[frame_id] = {
            'names': item['name'][valid],
            'boxes_lidar': item['boxes_lidar'][valid],
            'pred_labels': item['pred_labels'][valid]
        }
    
    frames = [os.path.splitext(os.path.basename(f))[0] for f in pcd_files]
    total_filled = 0
    os.makedirs(out_dir, exist_ok=True)
    
    for i, frame_id in enumerate(tqdm(frames, desc='Generating HEURISTIC Occupancy (U5)')):
        curr_pose = pose_dict[frame_id]['matrix']
        curr_pose_inv = np.linalg.inv(curr_pose)
        curr_boxes = box_dict.get(frame_id, {'names': [], 'boxes_lidar': []})
        
        min_b = np.array(GT_BOUNDS[:3])
        max_b = np.array(GT_BOUNDS[3:])
        
        pcd = o3d.io.read_point_cloud(pcd_files[i])
        curr_pts = np.asarray(pcd.points)
        pcs = [curr_pts]
        
        for j in range(1, num_sweeps + 1):
            if i - j < 0: break
            prev_id = frames[i - j]
            prev_pose = pose_dict[prev_id]['matrix']
            prev_pcd = o3d.io.read_point_cloud(pcd_files[i - j])
            prev_pts = np.asarray(prev_pcd.points)
            prev_boxes = box_dict.get(prev_id, {'boxes_lidar': []})['boxes_lidar']
            if len(prev_boxes) > 0:
                mask = points_in_boxes(prev_pts, prev_boxes)
                prev_pts = prev_pts[~mask]
            if len(prev_pts) > 0:
                p_homo = np.hstack((prev_pts, np.ones((len(prev_pts), 1))))
                transformed = (curr_pose_inv @ prev_pose @ p_homo.T).T[:, :3]
                pcs.append(transformed)
        
        for j in range(1, num_sweeps + 1):
            if i + j >= len(frames): break
            next_id = frames[i + j]
            next_pose = pose_dict[next_id]['matrix']
            next_pcd = o3d.io.read_point_cloud(pcd_files[i + j])
            next_pts = np.asarray(next_pcd.points)
            next_boxes = box_dict.get(next_id, {'boxes_lidar': []})['boxes_lidar']
            if len(next_boxes) > 0:
                mask = points_in_boxes(next_pts, next_boxes)
                next_pts = next_pts[~mask]
            if len(next_pts) > 0:
                p_homo = np.hstack((next_pts, np.ones((len(next_pts), 1))))
                transformed = (curr_pose_inv @ next_pose @ p_homo.T).T[:, :3]
                pcs.append(transformed)
        
        all_pts = np.vstack(pcs)
        
        mask = np.all((all_pts >= min_b) & (all_pts < max_b), axis=1)
        valid_xyz = all_pts[mask]
        idxs = np.clip(((valid_xyz - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
        occ = np.ones(GT_GRID, dtype=np.uint8) * LBL_FREE
        
        # U5 地面在 z ≈ 0m，z < 0.2m → 道路
        z_real = valid_xyz[:, 2]
        lbl = np.full(len(z_real), LBL_MANMADE, dtype=np.uint8)
        lbl[z_real < 0.2] = LBL_ROAD
        occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = lbl
        
        if len(curr_boxes['names']) > 0:
            occ, filled = fill_box_interior(occ, curr_boxes['boxes_lidar'], curr_boxes['names'], GT_BOUNDS, GT_VOXEL, GT_GRID)
            total_filled += filled
        
        save_path = os.path.join(out_dir, frame_id)
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(os.path.join(save_path, 'labels.npz'), semantics=occ)
    
    print(f"\n[HEURISTIC Pipeline Complete] 總共有 {total_filled} 個動態物件 Voxel 被 3D Box 霸道填滿！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='kiss_icp_u5')
    parser.add_argument('--scene', default='test_2026-03-16-11-31-09')
    parser.add_argument('--sweeps', type=int, default=40)
    parser.add_argument('--mode', default='all', choices=['raw', 'heuristic', 'all'],
                        help='raw=純原始, heuristic=高度判斷, all=兩種都跑')
    parser.add_argument('--data_root', default='/home/t113c52027/t113c52027/occ_gt_v2/data/u5')
    parser.add_argument('--out_root', default=os.path.join(os.path.dirname(__file__), 'output'))
    args = parser.parse_args()

    scene_path = os.path.join(args.data_root, args.scene)
    import importlib
    backend_mod = importlib.import_module(f'pose_backends.{args.backend}')
    pose_dict = backend_mod.get_pose_dict(scene_path)
    
    scene_out = os.path.join(args.out_root, args.scene)
    
    if args.mode in ('raw', 'all'):
        out_dir = os.path.join(scene_out, 'raw')
        generate_occupancy_raw(scene_path, pose_dict, args.sweeps, out_dir=out_dir)
    
    if args.mode in ('heuristic', 'all'):
        out_dir_heur = os.path.join(scene_out, 'heuristic')
        generate_occupancy_heuristic(scene_path, pose_dict, args.sweeps, out_dir=out_dir_heur)

if __name__ == '__main__':
    main()
