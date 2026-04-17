#!/usr/bin/env python3
"""
Generate Occupancy in CVPR2023 GT-compatible format (200x200x16 dense grid)
V2: Fills interior of 3D bounding boxes to match GT's solid volumes

Key improvement over V1:
  After LiDAR-based voxelization, use NuScenes 3D bounding box annotations
  to fill the interior of dynamic objects (car, truck, pedestrian, etc.)
  Only fills voxels that are currently FREE (17), won't overwrite existing labels.
"""

import os
import argparse
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion
from tqdm import tqdm

# --- Label Mapping: NuScenes lidarseg (0-31) -> Occ3D (0-16) ---
LIDARSEG_TO_OCC3D = np.zeros(32, dtype=np.uint8)
LIDARSEG_TO_OCC3D[0] = 0; LIDARSEG_TO_OCC3D[1] = 0
LIDARSEG_TO_OCC3D[5] = 0; LIDARSEG_TO_OCC3D[7] = 0; LIDARSEG_TO_OCC3D[8] = 0
LIDARSEG_TO_OCC3D[10] = 0; LIDARSEG_TO_OCC3D[11] = 0; LIDARSEG_TO_OCC3D[13] = 0
LIDARSEG_TO_OCC3D[19] = 0; LIDARSEG_TO_OCC3D[20] = 0; LIDARSEG_TO_OCC3D[29] = 0; LIDARSEG_TO_OCC3D[31] = 0
LIDARSEG_TO_OCC3D[9] = 1; LIDARSEG_TO_OCC3D[14] = 2
LIDARSEG_TO_OCC3D[15] = 3; LIDARSEG_TO_OCC3D[16] = 3
LIDARSEG_TO_OCC3D[17] = 4; LIDARSEG_TO_OCC3D[18] = 5; LIDARSEG_TO_OCC3D[21] = 6
LIDARSEG_TO_OCC3D[2] = 7; LIDARSEG_TO_OCC3D[3] = 7; LIDARSEG_TO_OCC3D[4] = 7; LIDARSEG_TO_OCC3D[6] = 7
LIDARSEG_TO_OCC3D[12] = 8; LIDARSEG_TO_OCC3D[22] = 9; LIDARSEG_TO_OCC3D[23] = 10
LIDARSEG_TO_OCC3D[24] = 11; LIDARSEG_TO_OCC3D[25] = 12; LIDARSEG_TO_OCC3D[26] = 13
LIDARSEG_TO_OCC3D[27] = 14; LIDARSEG_TO_OCC3D[28] = 15; LIDARSEG_TO_OCC3D[30] = 16

# --- NuScenes category name -> Occ3D label ---
CATEGORY_TO_OCC3D = {
    'human.pedestrian.adult': 7, 'human.pedestrian.child': 7,
    'human.pedestrian.construction_worker': 7, 'human.pedestrian.police_officer': 7,
    'movable_object.barrier': 1, 'movable_object.trafficcone': 8,
    'vehicle.bicycle': 2, 'vehicle.bus.bendy': 3, 'vehicle.bus.rigid': 3,
    'vehicle.car': 4, 'vehicle.construction': 5, 'vehicle.motorcycle': 6,
    'vehicle.trailer': 9, 'vehicle.truck': 10,
}

# --- GT-compatible parameters ---
GT_BOUNDS = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
GT_VOXEL = 0.4
GT_GRID = (200, 200, 16)

# ========================
# Helper functions (same as V1)
# ========================

def translate(points, x):
    for i in range(3):
        points[i, :] = points[i, :] + x[i]
    return points

def rotate(points, rot_matrix, center=None):
    if center is not None:
        points[:3, :] = np.dot(rot_matrix, points[:3, :] - center[:, None]) + center[:, None]
    else:
        points[:3, :] = np.dot(rot_matrix, points[:3, :])
    return points

def transform(points, rotate_matrix, translation_matrix, inverse=False):
    if not inverse:
        points = rotate(points, rotate_matrix)
        points = translate(points, translation_matrix)
    else:
        points = translate(points, -translation_matrix)
        points = rotate(points, np.linalg.inv(rotate_matrix))
    return points

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
        'pc': pc, 'token': sample['token'],
        'cs_record': cs_record, 'pose_record': pose_record,
        'lidarseg': points_label, 'boxes': boxes,
        'instance_tokens': instance_tokens
    }

def prev2ego(points, prev_info, ego_info):
    prev_cs = prev_info['cs_record']
    prev_pose = prev_info['pose_record']
    ego_cs = ego_info['cs_record']
    ego_pose = ego_info['pose_record']
    points = transform(points, Quaternion(prev_cs['rotation']).rotation_matrix, np.array(prev_cs['translation']))
    points = transform(points, Quaternion(prev_pose['rotation']).rotation_matrix, np.array(prev_pose['translation']))
    points = transform(points, Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']), inverse=True)
    points = transform(points, Quaternion(ego_cs['rotation']).rotation_matrix, np.array(ego_cs['translation']), inverse=True)
    return points

def keyframe_align(prev_info, ego_info):
    pc = prev_info['pc'].points.copy()
    seg = prev_info['lidarseg'].copy()
    ego_mask = (seg == 31)
    pc = pc[:, ~ego_mask]
    seg = seg[~ego_mask]
    static_mask = (seg >= 24) & (seg <= 30)
    static_points = pc[:, static_mask]
    static_seg = seg[static_mask]
    static_points = prev2ego(static_points, prev_info, ego_info)
    pcs = [static_points]
    segs = [static_seg]
    for i, box in enumerate(prev_info['boxes']):
        inst_token = prev_info['instance_tokens'][i]
        if inst_token not in ego_info['instance_tokens']:
            continue
        box_mask = points_in_box(box, prev_info['pc'].points[:3, :])
        if np.sum(box_mask) == 0: continue
        box_p = prev_info['pc'].points[:, box_mask].copy()
        box_s = prev_info['lidarseg'][box_mask].copy()
        prev_center = box.center
        prev_rot = box.rotation_matrix
        cur_idx = ego_info['instance_tokens'].index(inst_token)
        cur_box = ego_info['boxes'][cur_idx]
        cur_center = cur_box.center
        cur_rot = cur_box.rotation_matrix
        box_p = rotate(box_p, np.linalg.inv(prev_rot), center=prev_center)
        box_p = translate(box_p, cur_center - prev_center)
        box_p = rotate(box_p, cur_rot, center=cur_center)
        pcs.append(box_p)
        segs.append(box_s)
    return np.concatenate(pcs, axis=-1), np.concatenate(segs)

# ========================
# V2: 3D Box Interior Filling
# ========================

def fill_box_interior(occ, boxes, cs_record):
    """
    Fill interior of 3D bounding boxes in the occupancy grid.
    boxes: list of NuScenes Box objects (in LiDAR frame)
    cs_record: calibrated sensor record (for LiDAR -> Ego transform)
    Only fills voxels that are currently FREE (17).
    """
    min_bound = np.array(GT_BOUNDS[:3])
    R_l2e = Quaternion(cs_record['rotation']).rotation_matrix
    t_l2e = np.array(cs_record['translation'])
    
    filled_count = 0
    
    for box in boxes:
        # Get Occ3D label for this category
        occ3d_label = CATEGORY_TO_OCC3D.get(box.name, None)
        if occ3d_label is None:
            continue  # Skip categories not in Occ3D (e.g., stroller, debris)
        
        # Box parameters in LiDAR frame
        center_lidar = box.center  # (3,)
        w, l, h = box.wlh  # width, length, height
        rot_matrix = box.rotation_matrix  # (3, 3)
        
        # Transform box center to Ego frame
        center_ego = R_l2e @ center_lidar + t_l2e
        # Rotation in Ego frame
        rot_ego = R_l2e @ rot_matrix
        
        # NuScenes box-local frame convention:
        #   x-axis = length (l), y-axis = width (w), z-axis = height (h)
        # So half extents in box-local [x, y, z] = [l/2, w/2, h/2]
        half_ext = np.array([l / 2, w / 2, h / 2])
        
        # 8 corners of OBB in box-local frame
        corners_local = np.array([
            [-half_ext[0], -half_ext[1], -half_ext[2]],
            [ half_ext[0], -half_ext[1], -half_ext[2]],
            [-half_ext[0],  half_ext[1], -half_ext[2]],
            [ half_ext[0],  half_ext[1], -half_ext[2]],
            [-half_ext[0], -half_ext[1],  half_ext[2]],
            [ half_ext[0], -half_ext[1],  half_ext[2]],
            [-half_ext[0],  half_ext[1],  half_ext[2]],
            [ half_ext[0],  half_ext[1],  half_ext[2]],
        ])
        # Transform corners to Ego frame
        corners_ego = (rot_ego @ corners_local.T).T + center_ego
        
        # AABB in Ego frame
        aabb_min = corners_ego.min(axis=0)
        aabb_max = corners_ego.max(axis=0)
        
        # Convert AABB to voxel index range
        idx_min = np.floor((aabb_min - min_bound) / GT_VOXEL).astype(int)
        idx_max = np.ceil((aabb_max - min_bound) / GT_VOXEL).astype(int)
        
        # Clip to grid bounds
        idx_min = np.clip(idx_min, 0, np.array(GT_GRID) - 1)
        idx_max = np.clip(idx_max, 0, np.array(GT_GRID) - 1)
        
        # Generate candidate voxel centers in the AABB region
        xs = np.arange(idx_min[0], idx_max[0] + 1)
        ys = np.arange(idx_min[1], idx_max[1] + 1)
        zs = np.arange(idx_min[2], idx_max[2] + 1)
        
        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            continue
        
        # Create grid of voxel centers
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
        voxel_indices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        voxel_centers = voxel_indices * GT_VOXEL + min_bound + GT_VOXEL / 2  # (M, 3)
        
        # Check which voxel centers are inside the oriented bounding box
        # Transform voxel centers to box-local frame
        relative = voxel_centers - center_ego  # (M, 3)
        local_coords = (rot_ego.T @ relative.T).T  # (M, 3) in box-local frame
        
        # Inside check: [x, y, z] within [-l/2, l/2] x [-w/2, w/2] x [-h/2, h/2]
        inside = np.all(np.abs(local_coords) <= half_ext, axis=1)
        
        # Fill only FREE voxels
        inside_indices = voxel_indices[inside]
        for idx in inside_indices:
            if occ[idx[0], idx[1], idx[2]] == 17:  # Only fill free voxels
                occ[idx[0], idx[1], idx[2]] = occ3d_label
                filled_count += 1
    
    return occ, filled_count

# ========================
# Occupancy generation
# ========================

def generate_gt_format_occupancy(nusc, sample, num_sweeps=10):
    """Generate occupancy with V2 box filling"""
    
    curr_info = get_frame_info(sample, nusc)
    pcs = [curr_info['pc'].points]
    segs = [curr_info['lidarseg']]
    
    # Past frames
    prev_sample = sample
    for _ in range(num_sweeps):
        if prev_sample['prev'] == '': break
        prev_sample = nusc.get('sample', prev_sample['prev'])
        prev_info = get_frame_info(prev_sample, nusc)
        p, s = keyframe_align(prev_info, curr_info)
        pcs.append(p)
        segs.append(s)
    
    # Future frames
    next_sample = sample
    for _ in range(num_sweeps):
        if next_sample['next'] == '': break
        next_sample = nusc.get('sample', next_sample['next'])
        next_info = get_frame_info(next_sample, nusc)
        p, s = keyframe_align(next_info, curr_info)
        pcs.append(p)
        segs.append(s)
    
    # Concatenate
    all_pts = np.concatenate(pcs, axis=-1)
    all_lbl = np.concatenate(segs)
    
    # LiDAR -> Ego
    cs = curr_info['cs_record']
    R = Quaternion(cs['rotation']).rotation_matrix
    t = np.array(cs['translation'])
    xyz_ego = (R @ all_pts[:3, :] + t.reshape(3, 1)).T
    
    # Voxelization
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
    
    # *** V2: Fill interior of 3D bounding boxes ***
    occ, filled = fill_box_interior(occ, curr_info['boxes'], curr_info['cs_record'])
    
    return occ, filled

# ========================
# Main
# ========================

def main():
    parser = argparse.ArgumentParser(description="Generate CVPR2023 GT-compatible Occupancy (V2 with box filling)")
    parser.add_argument('--dataroot', default='/data1/nuscenes_occ/')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--out_root', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--scene_name', default='scene-0061')
    parser.add_argument('--num_sweeps', type=int, default=10)
    args = parser.parse_args()
    
    print(f"Loading NuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    
    scene = None
    for s in nusc.scene:
        if s['name'] == args.scene_name:
            scene = s
            break
    
    if scene is None:
        print(f"Scene {args.scene_name} not found!")
        return
    
    print(f"Processing {args.scene_name} ({scene['nbr_samples']} frames) [V2: with box filling]...")
    
    sample_token = scene['first_sample_token']
    frame_idx = 0
    total_filled = 0
    
    pbar = tqdm(total=scene['nbr_samples'], desc="Generating occupancy")
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        occ, filled = generate_gt_format_occupancy(nusc, sample, args.num_sweeps)
        total_filled += filled
        
        out_dir = os.path.join(args.out_root, args.scene_name, sample_token)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "labels.npz"), semantics=occ)
        
        sample_token = sample['next']
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    print(f"Done! Generated {frame_idx} frames to {args.out_root}")
    print(f"Total voxels filled by box interior: {total_filled} (avg {total_filled/frame_idx:.0f}/frame)")

if __name__ == "__main__":
    main()
