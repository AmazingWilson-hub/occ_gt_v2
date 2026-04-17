#!/usr/bin/env python3
"""
Generate Occupancy in GT-compatible format (200x200x16 dense grid)
For mIoU evaluation against CVPR2023 Occ3D GT
"""

import os
import argparse
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion
from tqdm import tqdm

# --- Label Mapping: NuScenes lidarseg (0-31) -> Occ3D evaluation (0-16) ---
# Based on official NuScenes lidarseg evaluation table:
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/README.md
LIDARSEG_TO_OCC3D = np.zeros(32, dtype=np.uint8)

# Class 0 = void/ignore (will be ignored in eval, but we set to 0 = others for now)
LIDARSEG_TO_OCC3D[0] = 0   # noise -> void/ignore -> 0
LIDARSEG_TO_OCC3D[1] = 0   # animal -> void/ignore -> 0
LIDARSEG_TO_OCC3D[5] = 0   # human.pedestrian.personal_mobility -> void/ignore -> 0
LIDARSEG_TO_OCC3D[7] = 0   # human.pedestrian.stroller -> void/ignore -> 0
LIDARSEG_TO_OCC3D[8] = 0   # human.pedestrian.wheelchair -> void/ignore -> 0
LIDARSEG_TO_OCC3D[10] = 0  # movable_object.debris -> void/ignore -> 0
LIDARSEG_TO_OCC3D[11] = 0  # movable_object.pushable_pullable -> void/ignore -> 0
LIDARSEG_TO_OCC3D[13] = 0  # static_object.bicycle_rack -> void/ignore -> 0
LIDARSEG_TO_OCC3D[19] = 0  # vehicle.emergency.ambulance -> void/ignore -> 0
LIDARSEG_TO_OCC3D[20] = 0  # vehicle.emergency.police -> void/ignore -> 0
LIDARSEG_TO_OCC3D[29] = 0  # static.other -> void/ignore -> 0
LIDARSEG_TO_OCC3D[31] = 0  # vehicle.ego -> void/ignore -> 0

# Actual mapped classes (1-16)
LIDARSEG_TO_OCC3D[9] = 1   # movable_object.barrier -> barrier
LIDARSEG_TO_OCC3D[14] = 2  # vehicle.bicycle -> bicycle
LIDARSEG_TO_OCC3D[15] = 3  # vehicle.bus.bendy -> bus
LIDARSEG_TO_OCC3D[16] = 3  # vehicle.bus.rigid -> bus
LIDARSEG_TO_OCC3D[17] = 4  # vehicle.car -> car
LIDARSEG_TO_OCC3D[18] = 5  # vehicle.construction -> construction_vehicle
LIDARSEG_TO_OCC3D[21] = 6  # vehicle.motorcycle -> motorcycle
LIDARSEG_TO_OCC3D[2] = 7   # human.pedestrian.adult -> pedestrian
LIDARSEG_TO_OCC3D[3] = 7   # human.pedestrian.child -> pedestrian
LIDARSEG_TO_OCC3D[4] = 7   # human.pedestrian.construction_worker -> pedestrian
LIDARSEG_TO_OCC3D[6] = 7   # human.pedestrian.police_officer -> pedestrian
LIDARSEG_TO_OCC3D[12] = 8  # movable_object.trafficcone -> traffic_cone
LIDARSEG_TO_OCC3D[22] = 9  # vehicle.trailer -> trailer
LIDARSEG_TO_OCC3D[23] = 10 # vehicle.truck -> truck
LIDARSEG_TO_OCC3D[24] = 11 # flat.driveable_surface -> driveable_surface
LIDARSEG_TO_OCC3D[25] = 12 # flat.other -> other_flat
LIDARSEG_TO_OCC3D[26] = 13 # flat.sidewalk -> sidewalk
LIDARSEG_TO_OCC3D[27] = 14 # flat.terrain -> terrain
LIDARSEG_TO_OCC3D[28] = 15 # static.manmade -> manmade
LIDARSEG_TO_OCC3D[30] = 16 # static.vegetation -> vegetation

# --- GT-compatible parameters ---
GT_BOUNDS = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
GT_VOXEL = 0.4
GT_GRID = (200, 200, 16)

def transform(points, R, t, inverse=False):
    if inverse:
        return R.T @ (points - t.reshape(3, 1))
    return R @ points + t.reshape(3, 1)

def get_frame_info(sample, nusc):
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_path, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
    lidarseg_file = os.path.join(nusc.dataroot, nusc.get('lidarseg', sample['data']['LIDAR_TOP'])['filename'])
    points_label = np.fromfile(lidarseg_file, dtype=np.uint8)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_rec['filename']))
    
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    instance_tokens = [nusc.get('sample_annotation', tok)['instance_token'] for tok in sample['anns']]
    
    return {
        'pc': pc,
        'token': sample['token'],
        'cs_record': cs_record,
        'pose_record': pose_record,
        'lidarseg': points_label,
        'boxes': boxes,
        'instance_tokens': instance_tokens
    }

def translate(points, x):
    """Translate points by x"""
    for i in range(3):
        points[i, :] = points[i, :] + x[i]
    return points

def rotate(points, rot_matrix, center=None):
    """Rotate points around center"""
    if center is not None:
        points[:3, :] = np.dot(rot_matrix, points[:3, :] - center[:, None]) + center[:, None]
    else:
        points[:3, :] = np.dot(rot_matrix, points[:3, :])
    return points

def transform_pts(points, rotate_matrix, translation_matrix, inverse=False):
    """Apply rotation and translation transform"""
    if not inverse:
        points = rotate(points, rotate_matrix)
        points = translate(points, translation_matrix)
    else:
        points = translate(points, -translation_matrix)
        points = rotate(points, np.linalg.inv(rotate_matrix))
    return points

def prev2ego(points, prev_info, ego_info):
    """Transform points from prev LiDAR frame to ego LiDAR frame"""
    prev_cs = prev_info['cs_record']
    prev_pose = prev_info['pose_record']
    ego_cs = ego_info['cs_record']
    ego_pose = ego_info['pose_record']

    # LiDAR -> Ego -> Global
    points = transform_pts(points, Quaternion(prev_cs['rotation']).rotation_matrix, 
                          np.array(prev_cs['translation']))
    points = transform_pts(points, Quaternion(prev_pose['rotation']).rotation_matrix, 
                          np.array(prev_pose['translation']))
    
    # Global -> Ego -> LiDAR
    points = transform_pts(points, Quaternion(ego_pose['rotation']).rotation_matrix, 
                          np.array(ego_pose['translation']), inverse=True)
    points = transform_pts(points, Quaternion(ego_cs['rotation']).rotation_matrix, 
                          np.array(ego_cs['translation']), inverse=True)
    return points

def keyframe_align(prev_info, ego_info):
    """Align prev frame to ego frame with proper static/dynamic separation"""
    pc = prev_info['pc'].points.copy()
    seg = prev_info['lidarseg'].copy()
    
    # Remove ego vehicle points
    ego_mask = (seg == 31)
    pc = pc[:, ~ego_mask]
    seg = seg[~ego_mask]
    
    # Track which points are handled by boxes
    handled_mask = np.zeros(pc.shape[1], dtype=bool)
    
    pcs = []
    segs = []
    
    # 1. Process dynamic objects in boxes FIRST
    for i, box in enumerate(prev_info['boxes']):
        inst_token = prev_info['instance_tokens'][i]
        if inst_token not in ego_info['instance_tokens']:
            continue
        
        # Get points in this box (from filtered pc, not original)
        box_mask = points_in_box(box, pc[:3, :])
        if box_mask.sum() == 0:
            continue
        
        # Mark these points as handled
        handled_mask |= box_mask
        
        box_p = pc[:, box_mask].copy()
        box_s = seg[box_mask].copy()
        
        # Dynamic alignment
        prev_center = box.center
        prev_rot = box.rotation_matrix
        
        cur_idx = ego_info['instance_tokens'].index(inst_token)
        cur_box = ego_info['boxes'][cur_idx]
        cur_center = cur_box.center
        cur_rot = cur_box.rotation_matrix
        
        # Transform: to local -> translate -> to new orientation
        box_p = rotate(box_p, np.linalg.inv(prev_rot), center=prev_center)
        box_p = translate(box_p, cur_center - prev_center)
        box_p = rotate(box_p, cur_rot, center=cur_center)
        
        pcs.append(box_p)
        segs.append(box_s)
    
    # 2. Process remaining points (static + dynamic not in any box) with prev2ego
    remaining_pc = pc[:, ~handled_mask].copy()
    remaining_seg = seg[~handled_mask]
    
    if remaining_pc.shape[1] > 0:
        remaining_pc = prev2ego(remaining_pc, prev_info, ego_info)
        pcs.append(remaining_pc)
        segs.append(remaining_seg)
    
    return np.concatenate(pcs, axis=-1), np.concatenate(segs)

def generate_gt_format_occupancy(nusc, sample, num_sweeps=10):
    """Generate occupancy in GT-compatible format (200x200x16 dense)"""
    
    curr_info = get_frame_info(sample, nusc)
    pcs = [curr_info['pc'].points]
    segs = [curr_info['lidarseg']]
    
    # Past frames
    prev_sample = sample
    for _ in range(num_sweeps):
        if prev_sample['prev'] == '':
            break
        prev_sample = nusc.get('sample', prev_sample['prev'])
        prev_info = get_frame_info(prev_sample, nusc)
        p, s = keyframe_align(prev_info, curr_info)
        pcs.append(p)
        segs.append(s)
    
    # Future frames
    next_sample = sample
    for _ in range(num_sweeps):
        if next_sample['next'] == '':
            break
        next_sample = nusc.get('sample', next_sample['next'])
        next_info = get_frame_info(next_sample, nusc)
        p, s = keyframe_align(next_info, curr_info)
        pcs.append(p)
        segs.append(s)
    
    # Concatenate (all points are now in current LiDAR frame)
    all_pts = np.concatenate(pcs, axis=-1)
    all_lbl = np.concatenate(segs)
    
    # *** CRITICAL: Transform from LiDAR to Ego coordinates ***
    # GT is annotated in Ego frame, not LiDAR frame
    cs = curr_info['cs_record']
    R = Quaternion(cs['rotation']).rotation_matrix
    t = np.array(cs['translation'])
    xyz_lidar = all_pts[:3, :]
    xyz_ego = R @ xyz_lidar + t.reshape(3, 1)
    
    # Voxelization (now in Ego frame)
    min_bound = np.array(GT_BOUNDS[:3])
    max_bound = np.array(GT_BOUNDS[3:])
    
    xyz = xyz_ego.T  # (N, 3)
    
    # Filter to bounds
    mask = np.all((xyz >= min_bound) & (xyz < max_bound), axis=1)
    xyz = xyz[mask]
    labels = all_lbl[mask]
    
    # Map labels to Occ3D format
    labels = LIDARSEG_TO_OCC3D[np.clip(labels, 0, 31)]
    
    # Quantize
    indices = ((xyz - min_bound) / GT_VOXEL).astype(int)
    indices = np.clip(indices, 0, np.array(GT_GRID) - 1)
    
    # Create dense grid (17 = empty/free)
    occ = np.ones(GT_GRID, dtype=np.uint8) * 17
    occ[indices[:, 0], indices[:, 1], indices[:, 2]] = labels
    
    return occ

def main():
    parser = argparse.ArgumentParser(description="Generate GT-compatible Occupancy for mIoU evaluation")
    parser.add_argument('--dataroot', default='/data1/nuscenes_occ/')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--out_root', default='/home/t113c52027/t113c52027/occ_gt_v2/occupancy_pipeline/pred_occ/')
    parser.add_argument('--scene_name', default='scene-0061')
    parser.add_argument('--num_sweeps', type=int, default=10, help="Number of frames to aggregate (past + future)")
    args = parser.parse_args()
    
    print(f"Loading NuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    
    # Find scene
    scene = None
    for s in nusc.scene:
        if s['name'] == args.scene_name:
            scene = s
            break
    
    if scene is None:
        print(f"Scene {args.scene_name} not found!")
        return
    
    print(f"Processing {args.scene_name}...")
    
    # Iterate samples
    sample_token = scene['first_sample_token']
    frame_idx = 0
    
    pbar = tqdm(total=scene['nbr_samples'], desc="Generating occupancy")
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        
        # Generate
        occ = generate_gt_format_occupancy(nusc, sample, args.num_sweeps)
        
        # Save
        out_dir = os.path.join(args.out_root, args.scene_name, sample_token)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "labels.npz"), semantics=occ)
        
        sample_token = sample['next']
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    print(f"Done! Generated {frame_idx} frames to {args.out_root}")

if __name__ == "__main__":
    main()
