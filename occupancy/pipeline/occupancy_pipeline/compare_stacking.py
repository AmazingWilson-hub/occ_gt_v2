#!/usr/bin/env python3
"""
Compare original vs new stacking logic side-by-side.
Outputs two PLY files for each frame so user can visually compare.
"""

import os
import sys
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion
import open3d as o3d
from tqdm import tqdm

# Add presentation_exporter to path for utils
sys.path.insert(0, '/data2/t113c52027/occ_gt_v2/tools/presentation_exporter')
from utils.points_process import translate, rotate, transform

# NuScenes colors (0-31)
NUSC_COLORS = np.array([
    [0, 0, 0], [70, 130, 180], [0, 0, 230], [135, 206, 235], [100, 149, 237],
    [219, 112, 147], [0, 0, 128], [240, 128, 128], [138, 43, 226], [112, 128, 144],
    [210, 105, 30], [105, 105, 105], [47, 79, 79], [188, 143, 143], [220, 20, 60],
    [255, 127, 80], [255, 69, 0], [255, 158, 0], [233, 150, 70], [255, 83, 0],
    [255, 215, 0], [255, 61, 99], [255, 140, 0], [255, 99, 71], [0, 207, 191],
    [175, 0, 75], [75, 0, 75], [112, 180, 60], [222, 184, 135], [255, 228, 196],
    [0, 175, 0], [255, 240, 245]
]) / 255.0


def get_frame_info(frame, nusc):
    sd_rec = nusc.get('sample_data', frame['data']['LIDAR_TOP'])
    lidar_path, boxes, _ = nusc.get_sample_data(frame['data']['LIDAR_TOP'])
    lidarseg_file = os.path.join(nusc.dataroot, nusc.get('lidarseg', frame['data']['LIDAR_TOP'])['filename'])
    points_label = np.fromfile(lidarseg_file, dtype=np.uint8)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_rec['filename']))
    
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    instance_tokens = [nusc.get('sample_annotation', tok)['instance_token'] for tok in frame['anns']]
    
    return {
        'pc': pc, 'token': frame['token'],
        'cs_record': cs_record, 'pose_record': pose_record,
        'lidarseg': points_label, 'boxes': boxes,
        'instance_tokens': instance_tokens
    }


def prev2ego_original(points, prev_frame_info, income_frame_info):
    """EXACT copy from generate_demo_scene.py"""
    prev_cs = prev_frame_info['cs_record']
    prev_pose = prev_frame_info['pose_record']
    ego_cs = income_frame_info['cs_record']
    ego_pose = income_frame_info['pose_record']

    # Lidar -> Ego -> Global
    points = transform(points, Quaternion(prev_cs['rotation']).rotation_matrix, np.array(prev_cs['translation']))
    points = transform(points, Quaternion(prev_pose['rotation']).rotation_matrix, np.array(prev_pose['translation']))
    
    # Global -> Ego -> Lidar
    points = transform(points, Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']), inverse=True)
    points = transform(points, Quaternion(ego_cs['rotation']).rotation_matrix, np.array(ego_cs['translation']), inverse=True)
    return points


def keyframe_align_ORIGINAL(prev_frame_info, ego_frame_info):
    """EXACT copy from generate_demo_scene.py (lines 109-168)"""
    pc = prev_frame_info['pc'].points.copy()
    seg = prev_frame_info['lidarseg'].copy()
    
    # 1. Static
    ego_mask = (seg == 31) 
    pc = pc[:, ~ego_mask]
    seg = seg[~ego_mask]
    
    static_mask = (seg >= 24) & (seg <= 30)
    
    # Align Static
    static_points = pc[:, static_mask]
    static_seg = seg[static_mask]
    static_points = prev2ego_original(static_points, prev_frame_info, ego_frame_info)
    
    pcs = [static_points]
    segs = [static_seg]
    
    # 2. Dynamic
    for i, box in enumerate(prev_frame_info['boxes']):
        inst_token = prev_frame_info['instance_tokens'][i]
        if inst_token not in ego_frame_info['instance_tokens']:
            continue
            
        box_mask = points_in_box(box, prev_frame_info['pc'].points[:3, :])
        if np.sum(box_mask) == 0: continue
        
        box_p = prev_frame_info['pc'].points[:, box_mask].copy()
        box_s = prev_frame_info['lidarseg'][box_mask].copy()
        
        prev_center = box.center
        prev_rot = box.rotation_matrix
        
        cur_idx = ego_frame_info['instance_tokens'].index(inst_token)
        cur_box = ego_frame_info['boxes'][cur_idx]
        cur_center = cur_box.center
        cur_rot = cur_box.rotation_matrix
        
        box_p = rotate(box_p, np.linalg.inv(prev_rot), center=prev_center)
        box_p = translate(box_p, cur_center - prev_center)
        box_p = rotate(box_p, cur_rot, center=cur_center)
        
        pcs.append(box_p)
        segs.append(box_s)
        
    return np.concatenate(pcs, axis=-1), np.concatenate(segs)


# ===================================================
# NEW VERSION (from generate_gt_format.py)
# ===================================================

def translate_new(points, x):
    for i in range(3):
        points[i, :] = points[i, :] + x[i]
    return points

def rotate_new(points, rot_matrix, center=None):
    if center is not None:
        points[:3, :] = np.dot(rot_matrix, points[:3, :] - center[:, None]) + center[:, None]
    else:
        points[:3, :] = np.dot(rot_matrix, points[:3, :])
    return points

def transform_pts_new(points, rotate_matrix, translation_matrix, inverse=False):
    if not inverse:
        points = rotate_new(points, rotate_matrix)
        points = translate_new(points, translation_matrix)
    else:
        points = translate_new(points, -translation_matrix)
        points = rotate_new(points, np.linalg.inv(rotate_matrix))
    return points

def prev2ego_new(points, prev_info, ego_info):
    prev_cs = prev_info['cs_record']
    prev_pose = prev_info['pose_record']
    ego_cs = ego_info['cs_record']
    ego_pose = ego_info['pose_record']

    points = transform_pts_new(points, Quaternion(prev_cs['rotation']).rotation_matrix, 
                          np.array(prev_cs['translation']))
    points = transform_pts_new(points, Quaternion(prev_pose['rotation']).rotation_matrix, 
                          np.array(prev_pose['translation']))
    
    points = transform_pts_new(points, Quaternion(ego_pose['rotation']).rotation_matrix, 
                          np.array(ego_pose['translation']), inverse=True)
    points = transform_pts_new(points, Quaternion(ego_cs['rotation']).rotation_matrix, 
                          np.array(ego_cs['translation']), inverse=True)
    return points

def keyframe_align_NEW(prev_info, ego_info):
    """Current version from generate_gt_format.py"""
    pc = prev_info['pc'].points.copy()
    seg = prev_info['lidarseg'].copy()
    
    ego_mask = (seg == 31)
    pc = pc[:, ~ego_mask]
    seg = seg[~ego_mask]
    
    handled_mask = np.zeros(pc.shape[1], dtype=bool)
    
    pcs = []
    segs_list = []
    
    for i, box in enumerate(prev_info['boxes']):
        inst_token = prev_info['instance_tokens'][i]
        if inst_token not in ego_info['instance_tokens']:
            continue
        
        box_mask = points_in_box(box, pc[:3, :])
        if box_mask.sum() == 0:
            continue
        
        handled_mask |= box_mask
        
        box_p = pc[:, box_mask].copy()
        box_s = seg[box_mask].copy()
        
        prev_center = box.center
        prev_rot = box.rotation_matrix
        
        cur_idx = ego_info['instance_tokens'].index(inst_token)
        cur_box = ego_info['boxes'][cur_idx]
        cur_center = cur_box.center
        cur_rot = cur_box.rotation_matrix
        
        box_p = rotate_new(box_p, np.linalg.inv(prev_rot), center=prev_center)
        box_p = translate_new(box_p, cur_center - prev_center)
        box_p = rotate_new(box_p, cur_rot, center=cur_center)
        
        pcs.append(box_p)
        segs_list.append(box_s)
    
    remaining_pc = pc[:, ~handled_mask].copy()
    remaining_seg = seg[~handled_mask]
    
    if remaining_pc.shape[1] > 0:
        remaining_pc = prev2ego_new(remaining_pc, prev_info, ego_info)
        pcs.append(remaining_pc)
        segs_list.append(remaining_seg)
    
    return np.concatenate(pcs, axis=-1), np.concatenate(segs_list)


def save_ply(xyz, labels, path):
    colors = NUSC_COLORS[np.clip(labels, 0, 31)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:3, :].T)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/data1/nuscenes_occ/')
    parser.add_argument('--scene_name', default='scene-0061')
    parser.add_argument('--num_sweeps', type=int, default=10)
    parser.add_argument('--sample_idx', type=int, default=0, help='Which sample in scene to use (0-based)')
    args = parser.parse_args()
    
    out_dir = '/home/t113c52027/t113c52027/occ_gt_v2/occupancy_pipeline/compare_output/'
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading NuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
    
    # Find scene
    scene = None
    for s in nusc.scene:
        if s['name'] == args.scene_name:
            scene = s
            break
    
    if scene is None:
        print(f"Scene {args.scene_name} not found!")
        return
    
    # Get the requested sample
    sample_token = scene['first_sample_token']
    for _ in range(args.sample_idx):
        sample = nusc.get('sample', sample_token)
        if sample['next'] == '':
            break
        sample_token = sample['next']
    
    sample = nusc.get('sample', sample_token)
    curr_info = get_frame_info(sample, nusc)
    print(f"Processing sample {args.sample_idx}, token: {sample_token}")
    
    # ========== ORIGINAL ==========
    pcs_orig = [curr_info['pc'].points.copy()]
    segs_orig = [curr_info['lidarseg'].copy()]
    
    prev_sample = sample
    for _ in range(args.num_sweeps):
        if prev_sample['prev'] == '':
            break
        prev_sample = nusc.get('sample', prev_sample['prev'])
        prev_info = get_frame_info(prev_sample, nusc)
        p, s = keyframe_align_ORIGINAL(prev_info, curr_info)
        pcs_orig.append(p)
        segs_orig.append(s)
    
    next_sample = sample
    for _ in range(args.num_sweeps):
        if next_sample['next'] == '':
            break
        next_sample = nusc.get('sample', next_sample['next'])
        next_info = get_frame_info(next_sample, nusc)
        p, s = keyframe_align_ORIGINAL(next_info, curr_info)
        pcs_orig.append(p)
        segs_orig.append(s)
    
    all_pts_orig = np.concatenate(pcs_orig, axis=-1)
    all_lbl_orig = np.concatenate(segs_orig)
    
    # ========== NEW ==========
    pcs_new = [curr_info['pc'].points.copy()]
    segs_new = [curr_info['lidarseg'].copy()]
    
    prev_sample = sample
    for _ in range(args.num_sweeps):
        if prev_sample['prev'] == '':
            break
        prev_sample = nusc.get('sample', prev_sample['prev'])
        prev_info = get_frame_info(prev_sample, nusc)
        p, s = keyframe_align_NEW(prev_info, curr_info)
        pcs_new.append(p)
        segs_new.append(s)
    
    next_sample = sample
    for _ in range(args.num_sweeps):
        if next_sample['next'] == '':
            break
        next_sample = nusc.get('sample', next_sample['next'])
        next_info = get_frame_info(next_sample, nusc)
        p, s = keyframe_align_NEW(next_info, curr_info)
        pcs_new.append(p)
        segs_new.append(s)
    
    all_pts_new = np.concatenate(pcs_new, axis=-1)
    all_lbl_new = np.concatenate(segs_new)
    
    # ========== SAVE ==========
    orig_path = os.path.join(out_dir, "original.ply")
    new_path = os.path.join(out_dir, "new_version.ply")
    
    print(f"Original: {all_pts_orig.shape[1]} points")
    print(f"New:      {all_pts_new.shape[1]} points")
    
    save_ply(all_pts_orig, all_lbl_orig, orig_path)
    save_ply(all_pts_new, all_lbl_new, new_path)
    
    print(f"\nSaved to {out_dir}:")
    print(f"  original.ply     - generate_demo_scene.py logic")
    print(f"  new_version.ply  - generate_gt_format.py logic")


if __name__ == "__main__":
    main()
