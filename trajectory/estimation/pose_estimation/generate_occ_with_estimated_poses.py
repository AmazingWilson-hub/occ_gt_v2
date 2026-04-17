#!/usr/bin/env python3
"""
Generate NuScenes Occupancy using Estimated Ego Poses (from Full Fusion)
instead of GT ego_pose. Then compare with GT-pose occupancy via mIoU.

Usage:
  python3 generate_occ_with_estimated_poses.py [--scene scene-0061]
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import argparse

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'occupancy', 'nuscenes', 'v1'))

DATAROOT = '/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ'

# --- Occ3D parameters ---
GT_BOUNDS = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
GT_VOXEL = 0.4
GT_GRID = (200, 200, 16)

# NuScenes lidarseg → Occ3D label mapping
LIDARSEG_TO_OCC3D = np.zeros(32, dtype=np.uint8)
LIDARSEG_TO_OCC3D[9] = 1; LIDARSEG_TO_OCC3D[14] = 2; LIDARSEG_TO_OCC3D[15] = 3
LIDARSEG_TO_OCC3D[16] = 4; LIDARSEG_TO_OCC3D[17] = 5; LIDARSEG_TO_OCC3D[18] = 6
LIDARSEG_TO_OCC3D[21] = 7; LIDARSEG_TO_OCC3D[2] = 8; LIDARSEG_TO_OCC3D[3] = 8
LIDARSEG_TO_OCC3D[4] = 8; LIDARSEG_TO_OCC3D[5] = 8; LIDARSEG_TO_OCC3D[6] = 9
LIDARSEG_TO_OCC3D[7] = 9; LIDARSEG_TO_OCC3D[8] = 9; LIDARSEG_TO_OCC3D[22] = 10
LIDARSEG_TO_OCC3D[23] = 11; LIDARSEG_TO_OCC3D[24] = 12; LIDARSEG_TO_OCC3D[25] = 13
LIDARSEG_TO_OCC3D[26] = 14; LIDARSEG_TO_OCC3D[27] = 14; LIDARSEG_TO_OCC3D[28] = 15
LIDARSEG_TO_OCC3D[30] = 16


def load_nuscenes():
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version='v1.0-mini', dataroot=DATAROOT)


def transform(points, R, t, inverse=False):
    if inverse:
        return R.T @ (points - t.reshape(3, 1))
    return R @ points + t.reshape(3, 1)


def quat_wxyz_to_rot(q):
    """Convert wxyz quaternion to rotation matrix"""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def get_frame_info(sample, nusc):
    """Get all info for a frame"""
    from nuscenes.utils.data_classes import LidarPointCloud
    
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidarseg_file = os.path.join(nusc.dataroot, nusc.get('lidarseg', sample['data']['LIDAR_TOP'])['filename'])
    points_label = np.fromfile(lidarseg_file, dtype=np.uint8)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_rec['filename']))
    
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    gt_pose = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    return {
        'pc': pc,
        'token': sample['token'],
        'cs_record': cs_record,
        'gt_pose': gt_pose,
        'utime': sd_rec['timestamp'],
        'lidarseg': points_label,
    }


def prev2ego_with_pose(points, prev_cs, prev_pose, ego_cs, ego_pose):
    """Transform points using provided pose (GT or estimated)"""
    # LiDAR → Ego → Global
    points = transform(points, quat_wxyz_to_rot(prev_cs['rotation']), np.array(prev_cs['translation']))
    points = transform(points, quat_wxyz_to_rot(prev_pose['rotation']), np.array(prev_pose['translation']))
    
    # Global → Ego → LiDAR
    points = transform(points, quat_wxyz_to_rot(ego_pose['rotation']), np.array(ego_pose['translation']), inverse=True)
    points = transform(points, quat_wxyz_to_rot(ego_cs['rotation']), np.array(ego_cs['translation']), inverse=True)
    return points


def generate_occupancy(nusc, scene_name, pose_dict, num_sweeps=10):
    """
    Generate occupancy for all samples in a scene.
    pose_dict: {utime: {'rotation': [w,x,y,z], 'translation': [x,y,z]}} 
               If None, use GT ego_pose.
    """
    scene = [s for s in nusc.scene if s['name'] == scene_name][0]
    
    results = []
    sample_token = scene['first_sample_token']
    sample_idx = 0
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        curr_info = get_frame_info(sample, nusc)
        
        # Use estimated or GT pose for current frame
        if pose_dict is not None:
            curr_pose = pose_dict[curr_info['utime']]
        else:
            curr_pose = curr_info['gt_pose']
        
        pcs = [curr_info['pc'].points]
        segs = [curr_info['lidarseg']]
        
        # Past frames
        prev_sample = sample
        for _ in range(num_sweeps):
            if prev_sample['prev'] == '':
                break
            prev_sample = nusc.get('sample', prev_sample['prev'])
            prev_info = get_frame_info(prev_sample, nusc)
            
            if pose_dict is not None:
                prev_pose = pose_dict[prev_info['utime']]
            else:
                prev_pose = prev_info['gt_pose']
            
            # Transform prev points to current frame
            pts = prev2ego_with_pose(
                prev_info['pc'].points[:3, :],
                prev_info['cs_record'], prev_pose,
                curr_info['cs_record'], curr_pose
            )
            
            # Static only (remove dynamic objects — simplified: keep all)
            pcs.append(pts)
            segs.append(prev_info['lidarseg'])
        
        # Future frames
        next_sample = sample
        for _ in range(num_sweeps):
            if next_sample['next'] == '':
                break
            next_sample = nusc.get('sample', next_sample['next'])
            next_info = get_frame_info(next_sample, nusc)
            
            if pose_dict is not None:
                next_pose = pose_dict[next_info['utime']]
            else:
                next_pose = next_info['gt_pose']
            
            pts = prev2ego_with_pose(
                next_info['pc'].points[:3, :],
                next_info['cs_record'], next_pose,
                curr_info['cs_record'], curr_pose
            )
            pcs.append(pts)
            segs.append(next_info['lidarseg'])
        
        # Concatenate and voxelize
        all_pts = np.concatenate(pcs, axis=-1)
        all_lbl = np.concatenate(segs)
        
        # LiDAR → Ego
        cs = curr_info['cs_record']
        R = quat_wxyz_to_rot(cs['rotation'])
        t = np.array(cs['translation'])
        xyz_ego = (R @ all_pts[:3, :] + t.reshape(3, 1)).T
        
        # Voxelize
        min_bound = np.array(GT_BOUNDS[:3])
        max_bound = np.array(GT_BOUNDS[3:])
        mask = np.all((xyz_ego >= min_bound) & (xyz_ego < max_bound), axis=1)
        xyz = xyz_ego[mask]
        labels = all_lbl[mask]
        labels = LIDARSEG_TO_OCC3D[np.clip(labels, 0, 31)]
        
        indices = ((xyz - min_bound) / GT_VOXEL).astype(int)
        indices = np.clip(indices, 0, np.array(GT_GRID) - 1)
        
        occ = np.ones(GT_GRID, dtype=np.uint8) * 17
        occ[indices[:, 0], indices[:, 1], indices[:, 2]] = labels
        
        results.append(occ)
        sample_token = sample['next'] if sample['next'] else None
        sample_idx += 1
    
    return results


def compute_miou(occ_est, occ_gt, num_classes=18):
    """Compute per-class IoU and mIoU between estimated and GT occupancy"""
    ious = []
    class_names = [
        'noise', 'barrier', 'bicycle', 'bus', 'car', 'const_veh',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'drv_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation', 'free'
    ]
    
    for c in range(num_classes):
        pred_c = (occ_est == c)
        gt_c = (occ_gt == c)
        intersection = np.sum(pred_c & gt_c)
        union = np.sum(pred_c | gt_c)
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(float('nan'))
    
    valid_ious = [x for x in ious if not np.isnan(x)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    
    return miou, ious, class_names


def run_full_fusion_poses(nusc, scene_name):
    """Run Full Fusion to get estimated poses"""
    from estimate_ego_pose import (
        load_canbus, get_scene_gt_poses, get_scene_lidar_paths,
        method_full_fusion, get_cs_transform
    )
    
    gt_poses = get_scene_gt_poses(nusc, scene_name)
    canbus_imu = load_canbus(scene_name, 'ms_imu')
    canbus_pose = load_canbus(scene_name, 'pose')
    lidar_infos = get_scene_lidar_paths(nusc, scene_name)
    cs_record = lidar_infos[0]['cs_record']
    
    print("Running Full Fusion pose estimation...")
    estimated_poses = method_full_fusion(lidar_infos, canbus_imu, canbus_pose, gt_poses, cs_record)
    
    # Build pose_dict: utime → {rotation, translation}
    pose_dict = {}
    for i, gt in enumerate(gt_poses):
        T = estimated_poses[i]
        R = Rotation.from_matrix(T[:3, :3])
        q = R.as_quat()  # scipy: xyzw
        pose_dict[gt['utime']] = {
            'rotation': [q[3], q[0], q[1], q[2]],  # convert to wxyz
            'translation': T[:3, 3].tolist()
        }
    
    return pose_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='scene-0061')
    parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(__file__), 'output'))
    args = parser.parse_args()
    
    print("Loading NuScenes...")
    nusc = load_nuscenes()
    
    # 1. Get Full Fusion estimated poses
    pose_dict = run_full_fusion_poses(nusc, args.scene)
    
    # 2. Generate occupancy with estimated poses
    print("\nGenerating occupancy with ESTIMATED poses...")
    occ_est_list = generate_occupancy(nusc, args.scene, pose_dict)
    
    # 3. Generate occupancy with GT poses
    print("Generating occupancy with GT poses...")
    occ_gt_list = generate_occupancy(nusc, args.scene, None)
    
    # 4. Compare
    print(f"\n{'='*60}")
    print(f"  Occupancy Comparison: Estimated vs GT Poses")
    print(f"  Scene: {args.scene}, {len(occ_est_list)} frames")
    print(f"{'='*60}")
    
    all_mious = []
    for i, (occ_e, occ_g) in enumerate(zip(occ_est_list, occ_gt_list)):
        miou, ious, names = compute_miou(occ_e, occ_g)
        all_mious.append(miou)
    
    avg_miou = np.mean(all_mious)
    print(f"\n  Per-frame mIoU: min={min(all_mious):.4f}, max={max(all_mious):.4f}")
    print(f"  Average mIoU across {len(all_mious)} frames: {avg_miou:.4f}")
    
    # Detailed class IoU for last frame
    miou, ious, names = compute_miou(occ_est_list[-1], occ_gt_list[-1])
    print(f"\n  Per-class IoU (last frame):")
    for name, iou in zip(names, ious):
        if not np.isnan(iou):
            print(f"    {name:<15s}: {iou:.4f}")
    
    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'miou_per_frame.npy'), np.array(all_mious))
    print(f"\n  Saved to {args.out_dir}/miou_per_frame.npy")


if __name__ == "__main__":
    main()
