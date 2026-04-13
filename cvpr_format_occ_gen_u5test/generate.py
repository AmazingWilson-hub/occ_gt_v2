#!/usr/bin/env python3
"""
cvpr_format_occ_gen_u5test — Semantic Label Pipeline
Uses pre-computed per-point semantic labels (.npy) from fine-tuned segmentation model.
Each .npy file contains uint8 labels (262144,) aligned 1-to-1 with the .pcd point cloud.

Label mapping (Occ3D classes, same as nuscenes):
  0: others, 1: barrier, 2: bicycle, 3: bus, 4: car, 5: construction_vehicle,
  6: motorcycle, 7: pedestrian, 8: traffic_cone, 9: trailer, 10: truck,
  11: driveable_surface, 12: other_flat, 13: sidewalk, 14: terrain,
  15: manmade, 16: vegetation, 17: free
"""

import os
import sys
import glob
import pickle
import argparse
import importlib
import numpy as np
import open3d as o3d
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from u5_box_utils import fill_box_interior, points_in_boxes

# U5: Ouster OS2
GT_BOUNDS = [-40.0, -40.0, -2.0, 40.0, 40.0, 6.0]
GT_VOXEL  = 0.4
GT_GRID   = (200, 200, 20)
LBL_FREE  = 17


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def generate_occupancy_semantic(dataroot, seg_root, pose_dict, num_sweeps=40, out_dir='output'):
    """
    Multi-sweep occupancy using per-point semantic labels from .npy files.
    Static points use labels from segmentation; dynamic objects use box filling.
    """
    pcd_files = sorted(glob.glob(os.path.join(dataroot, 'os_2_points', '*.pcd')))
    npy_files = sorted(glob.glob(os.path.join(seg_root, '*.npy')))

    if len(pcd_files) != len(npy_files):
        print(f"[WARN] pcd count ({len(pcd_files)}) != npy count ({len(npy_files)}), using min")
    n_frames = min(len(pcd_files), len(npy_files))

    box_data = load_pkl(os.path.join(dataroot, '3dbox_result.pkl'))
    box_dict = {}
    for item in box_data:
        frame_id = item['file_name'].replace('.pcd', '')
        valid = item['score'] >= 0.4
        box_dict[frame_id] = {
            'names':       item['name'][valid],
            'boxes_lidar': item['boxes_lidar'][valid],
        }

    frames = [os.path.splitext(os.path.basename(f))[0] for f in pcd_files[:n_frames]]
    min_b  = np.array(GT_BOUNDS[:3])
    max_b  = np.array(GT_BOUNDS[3:])
    total_filled = 0
    os.makedirs(out_dir, exist_ok=True)

    for i, frame_id in enumerate(tqdm(frames, desc='Generating SEMANTIC Occupancy (U5)')):
        curr_pose     = pose_dict[frame_id]['matrix']
        curr_pose_inv = np.linalg.inv(curr_pose)
        curr_boxes    = box_dict.get(frame_id, {'names': [], 'boxes_lidar': []})

        # Current frame: load xyz + labels
        pcd      = o3d.io.read_point_cloud(pcd_files[i])
        curr_pts = np.asarray(pcd.points)            # (N, 3)
        curr_lbl = np.load(npy_files[i]).astype(np.uint8)  # (N,)

        # Remove points inside dynamic boxes from static pool (will be box-filled later)
        if len(curr_boxes['boxes_lidar']) > 0:
            dyn_mask     = points_in_boxes(curr_pts, curr_boxes['boxes_lidar'])
            static_pts   = curr_pts[~dyn_mask]
            static_lbl   = curr_lbl[~dyn_mask]
        else:
            static_pts = curr_pts
            static_lbl = curr_lbl

        pcs  = [static_pts]
        lbls = [static_lbl]

        # Past sweeps
        for j in range(1, num_sweeps + 1):
            if i - j < 0:
                break
            prev_id   = frames[i - j]
            prev_pose = pose_dict[prev_id]['matrix']
            prev_pcd  = o3d.io.read_point_cloud(pcd_files[i - j])
            prev_pts  = np.asarray(prev_pcd.points)
            prev_lbl  = np.load(npy_files[i - j]).astype(np.uint8)

            # Remove dynamic points from previous frame
            prev_boxes = box_dict.get(prev_id, {'boxes_lidar': []})['boxes_lidar']
            if len(prev_boxes) > 0:
                dyn_mask = points_in_boxes(prev_pts, prev_boxes)
                prev_pts = prev_pts[~dyn_mask]
                prev_lbl = prev_lbl[~dyn_mask]

            if len(prev_pts) == 0:
                continue

            # Transform to current frame
            p_homo      = np.hstack((prev_pts, np.ones((len(prev_pts), 1))))
            transformed = (curr_pose_inv @ prev_pose @ p_homo.T).T[:, :3]
            pcs.append(transformed)
            lbls.append(prev_lbl)

        # Future sweeps
        for j in range(1, num_sweeps + 1):
            if i + j >= n_frames:
                break
            next_id   = frames[i + j]
            next_pose = pose_dict[next_id]['matrix']
            next_pcd  = o3d.io.read_point_cloud(pcd_files[i + j])
            next_pts  = np.asarray(next_pcd.points)
            next_lbl  = np.load(npy_files[i + j]).astype(np.uint8)

            next_boxes = box_dict.get(next_id, {'boxes_lidar': []})['boxes_lidar']
            if len(next_boxes) > 0:
                dyn_mask = points_in_boxes(next_pts, next_boxes)
                next_pts = next_pts[~dyn_mask]
                next_lbl = next_lbl[~dyn_mask]

            if len(next_pts) == 0:
                continue

            p_homo      = np.hstack((next_pts, np.ones((len(next_pts), 1))))
            transformed = (curr_pose_inv @ next_pose @ p_homo.T).T[:, :3]
            pcs.append(transformed)
            lbls.append(next_lbl)

        # Voxelization
        all_pts = np.vstack(pcs)
        all_lbl = np.concatenate(lbls)

        in_bounds = np.all((all_pts >= min_b) & (all_pts < max_b), axis=1)
        valid_xyz = all_pts[in_bounds]
        valid_lbl = all_lbl[in_bounds]

        idxs = np.clip(((valid_xyz - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
        occ  = np.ones(GT_GRID, dtype=np.uint8) * LBL_FREE
        occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = valid_lbl

        # Box fill for dynamic objects
        if len(curr_boxes['names']) > 0:
            occ, filled = fill_box_interior(occ, curr_boxes['boxes_lidar'],
                                            curr_boxes['names'], GT_BOUNDS, GT_VOXEL, GT_GRID)
            total_filled += filled

        save_path = os.path.join(out_dir, frame_id)
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(os.path.join(save_path, 'labels.npz'), semantics=occ)

    print(f"\n[SEMANTIC Pipeline Complete] Box-filled voxels: {total_filled}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',  default='kiss_icp_u5')
    parser.add_argument('--scene',    default='test_2026-03-23-10-42-37')
    parser.add_argument('--sweeps',   type=int, default=40)
    parser.add_argument('--data_root', default='/data2/t113c52027/occ_gt_v2/data/u5')
    parser.add_argument('--seg_root',
                        default='/home/t113c52027/t113c52027/seg_lidar/output/test_2026-03-23-10-42-37')
    parser.add_argument('--out_root', default=os.path.join(os.path.dirname(__file__), 'output'))
    args = parser.parse_args()

    scene_path = os.path.join(args.data_root, args.scene)

    backend_mod = importlib.import_module(f'pose_backends.{args.backend}')
    pose_dict   = backend_mod.get_pose_dict(scene_path)

    out_dir = os.path.join(args.out_root, args.scene, 'semantic')
    generate_occupancy_semantic(scene_path, args.seg_root, pose_dict,
                                num_sweeps=args.sweeps, out_dir=out_dir)
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()
