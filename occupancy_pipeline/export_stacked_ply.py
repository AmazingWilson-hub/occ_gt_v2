#!/usr/bin/env python3
"""
Export stacked point cloud as PLY file from the GT-format occupancy generation pipeline
"""

import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion
import open3d as o3d

# Occ3D colors for 17 classes (0-16) + free (17)
OCC3D_COLORS = np.array([
    [0, 0, 0],        # 0: others (black)
    [255, 120, 50],   # 1: barrier (orange)
    [255, 192, 203],  # 2: bicycle (pink)
    [255, 255, 0],    # 3: bus (yellow)
    [0, 150, 245],    # 4: car (blue)
    [0, 255, 255],    # 5: construction_vehicle (cyan)
    [200, 180, 0],    # 6: motorcycle (dark yellow)
    [255, 0, 0],      # 7: pedestrian (red)
    [255, 240, 150],  # 8: traffic_cone (light yellow)
    [135, 60, 0],     # 9: trailer (brown)
    [160, 32, 240],   # 10: truck (purple)
    [255, 0, 255],    # 11: driveable_surface (magenta)
    [139, 137, 137],  # 12: other_flat (gray)
    [75, 0, 75],      # 13: sidewalk (dark purple)
    [150, 240, 80],   # 14: terrain (light green)
    [230, 230, 250],  # 15: manmade (lavender)
    [0, 175, 0],      # 16: vegetation (green)
    [128, 128, 128],  # 17: free (gray) - usually not shown
]) / 255.0

# Label Mapping
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

GT_BOUNDS = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

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
        'pc': pc, 'token': sample['token'],
        'cs_record': cs_record, 'pose_record': pose_record,
        'lidarseg': points_label, 'boxes': boxes,
        'instance_tokens': instance_tokens
    }

def prev2ego(points, prev_info, ego_info):
    xyz = points[:3, :]
    xyz = transform(xyz, Quaternion(prev_info['cs_record']['rotation']).rotation_matrix, 
                   np.array(prev_info['cs_record']['translation']))
    xyz = transform(xyz, Quaternion(prev_info['pose_record']['rotation']).rotation_matrix,
                   np.array(prev_info['pose_record']['translation']))
    xyz = transform(xyz, Quaternion(ego_info['pose_record']['rotation']).rotation_matrix,
                   np.array(ego_info['pose_record']['translation']), inverse=True)
    xyz = transform(xyz, Quaternion(ego_info['cs_record']['rotation']).rotation_matrix,
                   np.array(ego_info['cs_record']['translation']), inverse=True)
    result = points.copy()
    result[:3, :] = xyz
    return result

def align_dynamic(prev_info, ego_info, points, labels):
    for idx in range(len(prev_info['boxes'])):
        inst_tok = prev_info['instance_tokens'][idx]
        if inst_tok not in ego_info['instance_tokens']:
            continue
        box_mask = points_in_box(prev_info['boxes'][idx], points[:3, :])
        if box_mask.sum() == 0:
            continue
        box_pts = points[:, box_mask].copy()
        box_lbl = labels[box_mask].copy()
        points = points[:, ~box_mask]
        labels = labels[~box_mask]
        prev_box = prev_info['boxes'][idx]
        tgt_idx = ego_info['instance_tokens'].index(inst_tok)
        tgt_box = ego_info['boxes'][tgt_idx]
        box_pts[:3, :] = prev_box.rotation_matrix @ box_pts[:3, :] + prev_box.center.reshape(3, 1)
        box_pts[:3, :] = tgt_box.rotation_matrix.T @ (box_pts[:3, :] - tgt_box.center.reshape(3, 1))
        points = np.hstack([points, box_pts])
        labels = np.hstack([labels, box_lbl])
    return points, labels

def keyframe_align(prev_info, ego_info):
    pts = prev_info['pc'].points.copy()
    lbl = prev_info['lidarseg'].copy()
    pts, lbl = align_dynamic(prev_info, ego_info, pts, lbl)
    pts = prev2ego(pts, prev_info, ego_info)
    return pts, lbl

def generate_stacked_ply(nusc, sample, num_sweeps=10):
    """Generate stacked point cloud for a single sample"""
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
    
    # Concatenate
    all_pts = np.concatenate(pcs, axis=-1)
    all_lbl = np.concatenate(segs)
    
    # Transform LiDAR -> Ego
    cs = curr_info['cs_record']
    R = Quaternion(cs['rotation']).rotation_matrix
    t = np.array(cs['translation'])
    xyz_ego = (R @ all_pts[:3, :] + t.reshape(3, 1)).T  # (N, 3)
    
    # Filter to GT bounds
    min_b = np.array(GT_BOUNDS[:3])
    max_b = np.array(GT_BOUNDS[3:])
    mask = np.all((xyz_ego >= min_b) & (xyz_ego < max_b), axis=1)
    xyz_ego = xyz_ego[mask]
    labels = all_lbl[mask]
    
    # Map labels and get colors
    labels_occ = LIDARSEG_TO_OCC3D[np.clip(labels, 0, 31)]
    colors = OCC3D_COLORS[labels_occ]
    
    return xyz_ego, colors

def main():
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/data1/nuscenes_occ/')
    parser.add_argument('--scene_name', default='scene-0061')
    parser.add_argument('--out_root', default='/home/t113c52027/t113c52027/occ_gt_v2/occupancy_pipeline/pred_occ/')
    parser.add_argument('--num_sweeps', type=int, default=10)
    args = parser.parse_args()
    
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
    
    print(f"Processing {args.scene_name} ({scene['nbr_samples']} frames)...")
    
    sample_token = scene['first_sample_token']
    frame_idx = 0
    
    pbar = tqdm(total=scene['nbr_samples'], desc="Generating PLY")
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        
        # Generate stacked point cloud
        xyz, colors = generate_stacked_ply(nusc, sample, args.num_sweeps)
        
        # Save PLY
        out_dir = os.path.join(args.out_root, args.scene_name, sample_token)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "stacked.ply")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(out_path, pcd)
        
        sample_token = sample['next']
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    print(f"Done! Generated {frame_idx} PLY files to {args.out_root}/{args.scene_name}/")

if __name__ == "__main__":
    main()
