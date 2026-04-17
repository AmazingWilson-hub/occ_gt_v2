#!/usr/bin/env python3
"""
Compare ORIGINAL vs NEW occupancy voxelization.
Outputs occupancy as PLY voxel centers for visual comparison.
"""

import os, sys
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion
import open3d as o3d

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

# Occ3D colors (0-17)
OCC3D_COLORS = np.array([
    [0, 0, 0],        # 0: others
    [255, 120, 50],   # 1: barrier
    [255, 192, 203],  # 2: bicycle
    [255, 255, 0],    # 3: bus
    [0, 150, 245],    # 4: car
    [0, 255, 255],    # 5: construction_vehicle
    [200, 180, 0],    # 6: motorcycle
    [255, 0, 0],      # 7: pedestrian
    [255, 240, 150],  # 8: traffic_cone
    [135, 60, 0],     # 9: trailer
    [160, 32, 240],   # 10: truck
    [255, 0, 255],    # 11: driveable_surface
    [139, 137, 137],  # 12: other_flat
    [75, 0, 75],      # 13: sidewalk
    [150, 240, 80],   # 14: terrain
    [230, 230, 250],  # 15: manmade
    [0, 175, 0],      # 16: vegetation
    [128, 128, 128],  # 17: free
]) / 255.0

# Label mapping
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

def prev2ego_orig(points, prev_frame_info, income_frame_info):
    prev_cs = prev_frame_info['cs_record']
    prev_pose = prev_frame_info['pose_record']
    ego_cs = income_frame_info['cs_record']
    ego_pose = income_frame_info['pose_record']
    points = transform(points, Quaternion(prev_cs['rotation']).rotation_matrix, np.array(prev_cs['translation']))
    points = transform(points, Quaternion(prev_pose['rotation']).rotation_matrix, np.array(prev_pose['translation']))
    points = transform(points, Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']), inverse=True)
    points = transform(points, Quaternion(ego_cs['rotation']).rotation_matrix, np.array(ego_cs['translation']), inverse=True)
    return points

def keyframe_align_ORIGINAL(prev_frame_info, ego_frame_info):
    pc = prev_frame_info['pc'].points.copy()
    seg = prev_frame_info['lidarseg'].copy()
    ego_mask = (seg == 31)
    pc = pc[:, ~ego_mask]
    seg = seg[~ego_mask]
    static_mask = (seg >= 24) & (seg <= 30)
    static_points = pc[:, static_mask]
    static_seg = seg[static_mask]
    static_points = prev2ego_orig(static_points, prev_frame_info, ego_frame_info)
    pcs = [static_points]
    segs = [static_seg]
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


def stack_frames(nusc, sample, num_sweeps, align_func):
    """Stack multi-frame using given alignment function"""
    curr_info = get_frame_info(sample, nusc)
    pcs = [curr_info['pc'].points.copy()]
    segs_list = [curr_info['lidarseg'].copy()]
    
    prev_sample = sample
    for _ in range(num_sweeps):
        if prev_sample['prev'] == '': break
        prev_sample = nusc.get('sample', prev_sample['prev'])
        prev_info = get_frame_info(prev_sample, nusc)
        p, s = align_func(prev_info, curr_info)
        pcs.append(p)
        segs_list.append(s)
    
    next_sample = sample
    for _ in range(num_sweeps):
        if next_sample['next'] == '': break
        next_sample = nusc.get('sample', next_sample['next'])
        next_info = get_frame_info(next_sample, nusc)
        p, s = align_func(next_info, curr_info)
        pcs.append(p)
        segs_list.append(s)
    
    return np.concatenate(pcs, axis=-1), np.concatenate(segs_list), curr_info


def voxelize_original(all_pts, all_lbl):
    """Original voxelization: 0.2m, [-60,60], sparse, raw labels"""
    min_bound = np.array([-60.0, -60.0, -5.0])
    max_bound = np.array([60.0, 60.0, 11.0])
    voxel_size = 0.2
    
    xyz = all_pts[:3, :].T
    mask = np.all((xyz >= min_bound) & (xyz < max_bound), axis=1)
    xyz = xyz[mask]
    labels = all_lbl[mask]
    
    indices = ((xyz - min_bound) / voxel_size).astype(int)
    _, uniq_idx = np.unique(indices, axis=0, return_index=True)
    final_indices = indices[uniq_idx]
    final_labels = labels[uniq_idx]
    
    # Convert voxel indices back to XYZ for PLY
    voxel_centers = final_indices * voxel_size + min_bound + voxel_size / 2
    return voxel_centers, final_labels


def voxelize_gt_format(all_pts, all_lbl, curr_info):
    """New GT-format voxelization: 0.4m, [-40,40], dense, Occ3D labels, Ego coords"""
    GT_BOUNDS = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    GT_VOXEL = 0.4
    GT_GRID = (200, 200, 16)
    
    # LiDAR -> Ego
    cs = curr_info['cs_record']
    R = Quaternion(cs['rotation']).rotation_matrix
    t = np.array(cs['translation'])
    xyz_ego = (R @ all_pts[:3, :] + t.reshape(3, 1)).T
    
    min_bound = np.array(GT_BOUNDS[:3])
    max_bound = np.array(GT_BOUNDS[3:])
    
    mask = np.all((xyz_ego >= min_bound) & (xyz_ego < max_bound), axis=1)
    xyz = xyz_ego[mask]
    labels = all_lbl[mask]
    
    # Map labels
    labels_mapped = LIDARSEG_TO_OCC3D[np.clip(labels, 0, 31)]
    
    # Voxelize
    indices = ((xyz - min_bound) / GT_VOXEL).astype(int)
    indices = np.clip(indices, 0, np.array(GT_GRID) - 1)
    
    occ = np.ones(GT_GRID, dtype=np.uint8) * 17
    occ[indices[:, 0], indices[:, 1], indices[:, 2]] = labels_mapped
    
    # Convert dense grid to PLY (only non-free voxels)
    occupied = np.argwhere(occ != 17)
    occ_labels = occ[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
    voxel_centers = occupied * GT_VOXEL + min_bound + GT_VOXEL / 2
    
    return voxel_centers, occ_labels


def save_ply_nusc(xyz, labels, path):
    """Save PLY with NuScenes raw colors"""
    colors = NUSC_COLORS[np.clip(labels, 0, 31)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)

def save_ply_occ3d(xyz, labels, path):
    """Save PLY with Occ3D colors"""
    colors = OCC3D_COLORS[np.clip(labels, 0, 17)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/data1/nuscenes_occ/')
    parser.add_argument('--scene_name', default='scene-0061')
    parser.add_argument('--num_sweeps', type=int, default=10)
    parser.add_argument('--sample_idx', type=int, default=5)
    args = parser.parse_args()
    
    out_dir = '/home/t113c52027/t113c52027/occ_gt_v2/occupancy_pipeline/compare_output/'
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading NuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
    
    scene = None
    for s in nusc.scene:
        if s['name'] == args.scene_name:
            scene = s
            break
    
    sample_token = scene['first_sample_token']
    for _ in range(args.sample_idx):
        sample = nusc.get('sample', sample_token)
        if sample['next'] == '': break
        sample_token = sample['next']
    
    sample = nusc.get('sample', sample_token)
    print(f"Processing sample {args.sample_idx}, token: {sample_token}")
    
    # Stack with original logic
    print("\n[1] Stacking with ORIGINAL logic...")
    all_pts, all_lbl, curr_info = stack_frames(nusc, sample, args.num_sweeps, keyframe_align_ORIGINAL)
    
    # Original voxelization (0.2m, LiDAR frame, raw labels)
    print("[2] Voxelizing ORIGINAL (0.2m, LiDAR, raw labels)...")
    vox_xyz_orig, vox_lbl_orig = voxelize_original(all_pts, all_lbl)
    save_ply_nusc(vox_xyz_orig, vox_lbl_orig, os.path.join(out_dir, "occ_original_0.2m.ply"))
    print(f"    -> {len(vox_lbl_orig)} voxels")
    
    # GT-format voxelization (0.4m, Ego frame, Occ3D labels)  
    print("[3] Voxelizing GT-FORMAT (0.4m, Ego, Occ3D labels)...")
    vox_xyz_gt, vox_lbl_gt = voxelize_gt_format(all_pts, all_lbl, curr_info)
    save_ply_occ3d(vox_xyz_gt, vox_lbl_gt, os.path.join(out_dir, "occ_gt_format_0.4m.ply"))
    print(f"    -> {len(vox_lbl_gt)} voxels")
    
    # Also save the GT itself for reference
    gt_path = f"/data1/nuscenes_occ/gts/{args.scene_name}/{sample_token}/labels.npz"
    if os.path.exists(gt_path):
        print("[4] Loading actual GT...")
        gt_data = np.load(gt_path)
        gt = gt_data['semantics']
        gt_occupied = np.argwhere(gt != 17)
        gt_labels = gt[gt_occupied[:, 0], gt_occupied[:, 1], gt_occupied[:, 2]]
        min_bound = np.array([-40.0, -40.0, -1.0])
        gt_centers = gt_occupied * 0.4 + min_bound + 0.2
        save_ply_occ3d(gt_centers, gt_labels, os.path.join(out_dir, "occ_GT_actual.ply"))
        print(f"    -> {len(gt_labels)} voxels")
    
    print(f"\nAll files saved to {out_dir}")
    print("  occ_original_0.2m.ply  - Original voxelization (0.2m, LiDAR coords, NuScenes colors)")
    print("  occ_gt_format_0.4m.ply - GT-format voxelization (0.4m, Ego coords, Occ3D colors)")
    print("  occ_GT_actual.ply      - Actual GT (for reference)")

if __name__ == "__main__":
    main()
