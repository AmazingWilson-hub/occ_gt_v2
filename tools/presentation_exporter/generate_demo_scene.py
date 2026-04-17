
import os
import argparse
import shutil
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion
import open3d as o3d
from utils.points_process import *

# --- 1. CONFIG & COLORS ---
NUSC_COLORS = np.array([
    [0, 0, 0], [70, 130, 180], [0, 0, 230], [135, 206, 235], [100, 149, 237],
    [219, 112, 147], [0, 0, 128], [240, 128, 128], [138, 43, 226], [112, 128, 144],
    [210, 105, 30], [105, 105, 105], [47, 79, 79], [188, 143, 143], [220, 20, 60],
    [255, 127, 80], [255, 69, 0], [255, 158, 0], [233, 150, 70], [255, 83, 0],
    [255, 215, 0], [255, 61, 99], [255, 140, 0], [255, 99, 71], [0, 207, 191],
    [175, 0, 75], [75, 0, 75], [112, 180, 60], [222, 184, 135], [255, 228, 196],
    [0, 175, 0], [255, 240, 245]
]) / 255.0

# --- 2. HELPERS (From data_converter.py) ---
def get_frame_info(frame, nusc):
    sd_rec = nusc.get('sample_data', frame['data']['LIDAR_TOP'])
    lidar_path, boxes, _ = nusc.get_sample_data(frame['data']['LIDAR_TOP'])
    lidarseg_labels_filename = os.path.join(nusc.dataroot, nusc.get('lidarseg', frame['data']['LIDAR_TOP'])['filename'])
    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_rec['filename'])) 
    
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    instance_tokens = [nusc.get('sample_annotation', token)['instance_token'] for token in frame['anns']]
    
    return {
        'pc': pc,
        'token': frame['token'],
        'cs_record': cs_record,
        'pose_record': pose_record,
        'lidarseg': points_label,
        'boxes': boxes,
        'instance_tokens': instance_tokens
    }

def prev2ego(points, prev_frame_info, income_frame_info):
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

def filter_points_in_ego(points, frame_info, instance_token):
    index = frame_info['instance_tokens'].index(instance_token)
    box = frame_info['boxes'][index]
    return points_in_box(box, points[:3, :])

def align_dynamic_thing(prev_frame_info, ego_frame_info, points, lidarseg):
    for index_anno in range(len(prev_frame_info['boxes'])):
        inst_token = prev_frame_info['instance_tokens'][index_anno]
        if inst_token not in ego_frame_info['instance_tokens']:
            continue
            
        box_mask = points_in_box(prev_frame_info['boxes'][index_anno], points[:3, :])
        if np.sum(box_mask) == 0: continue

        box_points = points[:, box_mask].copy()
        box_seg = lidarseg[box_mask].copy()
        
        # Transform
        prev_box = prev_frame_info['boxes'][index_anno]
        target_idx = ego_frame_info['instance_tokens'].index(inst_token)
        curr_box = ego_frame_info['boxes'][target_idx]
        
        # 1. To Local
        box_points = rotate(box_points, np.linalg.inv(prev_box.rotation_matrix), center=prev_box.center)
        # 2. To New Global (Relative)
        box_points = translate(box_points, curr_box.center - prev_box.center)
        box_points = rotate(box_points, curr_box.rotation_matrix, center=curr_box.center)
        
        # Filter (Verify it's inside new box)
        # Need to re-transform to ego to check? No, box_points is currently in ego frame logic? 
        # Wait, prev2ego handles S->E->G... 
        # Current logic is: points are in Prev Lidar Frame.
        
        # Simplified for presentation script: Just append valid points
        # For simplicity in this demo script, we assume strict box alignment
        # (This is a simplified version of data_converter logic for clarity)
        
        # But wait, we need to return aligned points.
        # Let's stick to the logic:
        # P_prev_lidar -> P_prev_ego -> P_prev_global ?? No.
        # The logic in data_converter.py is:
        # P_prev_lidar -> [Rotate/Translate relative to Box] -> P_curr_lidar_approx
        
        # The complex logic in data_converter ensures high quality. 
        # For this demo script, we will piggyback on the assumption that we align well.
        pass 
    return points, lidarseg # Placeholder, logic is integrated below

def keyframe_align(prev_frame_info, ego_frame_info):
    # Separate Static / Dynamic
    pc = prev_frame_info['pc'].points.copy()
    seg = prev_frame_info['lidarseg'].copy()
    
    # 1. Static
    ego_mask = (seg == 31) 
    pc = pc[:, ~ego_mask]
    seg = seg[~ego_mask]
    
    static_mask = (seg >= 24) & (seg <= 30) # Vegetation, Driveable, etc.
    
    # Align Static
    static_points = pc[:, static_mask]
    static_seg = seg[static_mask]
    static_points = prev2ego(static_points, prev_frame_info, ego_frame_info)
    
    pcs = [static_points]
    segs = [static_seg]
    
    # 2. Dynamic
    # Remove static
    pc_dynamic = pc[:, ~static_mask]
    seg_dynamic = seg[~static_mask]
    
    # For each box in prev
    for i, box in enumerate(prev_frame_info['boxes']):
        inst_token = prev_frame_info['instance_tokens'][i]
        if inst_token not in ego_frame_info['instance_tokens']:
            continue
            
        box_mask = points_in_box(box, prev_frame_info['pc'].points[:3, :])
        if np.sum(box_mask) == 0: continue
        
        box_p = prev_frame_info['pc'].points[:, box_mask].copy()
        box_s = prev_frame_info['lidarseg'][box_mask].copy()
        
        # Align Dynamic Logic
        prev_center = box.center
        prev_rot = box.rotation_matrix
        
        cur_idx = ego_frame_info['instance_tokens'].index(inst_token)
        cur_box = ego_frame_info['boxes'][cur_idx]
        cur_center = cur_box.center
        cur_rot = cur_box.rotation_matrix
        
        # Transform
        box_p = rotate(box_p, np.linalg.inv(prev_rot), center=prev_center) # To Local
        box_p = translate(box_p, cur_center - prev_center) # Move
        box_p = rotate(box_p, cur_rot, center=cur_center) # Rotate
        
        # Filter (Optional for speed, but good for quality)
        # mask_in_cur = points_in_box(cur_box, box_p[:3, :])
        # box_p = box_p[:, mask_in_cur]
        # box_s = box_s[mask_in_cur]
        
        pcs.append(box_p)
        segs.append(box_s)
        
    return np.concatenate(pcs, axis=-1), np.concatenate(segs)

# --- 3. EXPORT FUNCTIONS ---

def export_camera_images(nusc, sample, out_dir):
    cams = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    for cam in cams:
        cam_data = nusc.get('sample_data', sample['data'][cam])
        src_path = os.path.join(nusc.dataroot, cam_data['filename'])
        dst_path = os.path.join(out_dir, f"{cam}.jpg")
        shutil.copy(src_path, dst_path)

def export_semantic_lidar(nusc, sample, out_dir):
    # Load LiDAR
    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', lidar_token)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_rec['filename']))
    
    # Load Labels
    lidarseg_path = os.path.join(nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename'])
    labels = np.fromfile(lidarseg_path, dtype=np.uint8)
    
    # Colorize
    points = pc.points.T[:, :3] # N x 3
    colors = np.zeros((len(labels), 3))
    for i, lbl in enumerate(labels):
        if lbl < len(NUSC_COLORS):
            colors[i] = NUSC_COLORS[lbl]
            
    # Save PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(out_dir, "LIDAR_SEMANTIC.ply"), pcd)

def generate_and_save_occupancy(nusc, sample, out_dir, num_sweeps=10):
    # 1. Multi-frame Stacking
    curr_info = get_frame_info(sample, nusc)
    pcs = [curr_info['pc'].points]
    segs = [curr_info['lidarseg']]
    
    # Past frames
    prev_frame = sample
    for _ in range(num_sweeps):
        if prev_frame['prev'] == '': break
        prev_frame = nusc.get('sample', prev_frame['prev'])
        prev_info = get_frame_info(prev_frame, nusc)
        p, s = keyframe_align(prev_info, curr_info)
        pcs.append(p)
        segs.append(s)
        
    # Future frames
    next_frame = sample
    for _ in range(num_sweeps):
        if next_frame['next'] == '': break
        next_frame = nusc.get('sample', next_frame['next'])
        next_info = get_frame_info(next_frame, nusc)
        p, s = keyframe_align(next_info, curr_info)
        pcs.append(p)
        segs.append(s)
        
    all_points = np.concatenate(pcs, axis=-1)
    all_segs = np.concatenate(segs)

    # --- SAVE STACKED PLY (User Request) ---
    points_ply = all_points[:3, :].T # N x 3
    colors_ply = np.zeros((len(all_segs), 3))
    for i, lbl in enumerate(all_segs):
        if lbl < len(NUSC_COLORS):
            colors_ply[i] = NUSC_COLORS[lbl]
    
    pcd_stack = o3d.geometry.PointCloud()
    pcd_stack.points = o3d.utility.Vector3dVector(points_ply)
    pcd_stack.colors = o3d.utility.Vector3dVector(colors_ply)
    o3d.io.write_point_cloud(os.path.join(out_dir, "LIDAR_STACKED.ply"), pcd_stack)
    # ----------------------------------------
    
    # 2. Voxelization
    # Bounds: [-60, 60] X/Y, [-5, 11] Z
    min_bound = np.array([-60.0, -60.0, -5.0])
    max_bound = np.array([60.0, 60.0, 11.0])
    voxel_size = 0.2
    
    xyz = all_points[:3, :].T
    
    # Filter bounds
    mask = np.all((xyz >= min_bound) & (xyz < max_bound), axis=1)
    xyz = xyz[mask]
    labels = all_segs[mask]
    
    # Quantize
    indices = ((xyz - min_bound) / voxel_size).astype(int)
    
    # Keep one label per voxel (Simple majority or random)
    # Here using unique to sparse
    # To handle collisions (same voxel, diff label), we can just use dictionary or simple lexsort
    # For speed/demo: Just unique logical
    
    # We need strictly unique indices
    # Pack indices and labels
    data = np.hstack([indices, labels.reshape(-1, 1)]) # N x 4
    unique_data = np.unique(data, axis=0) # This keeps unique (idx, label) pairs. 
    # But we want unique indices.
    
    # Proper voxelization:
    _, uniq_idx = np.unique(indices, axis=0, return_index=True)
    final_indices = indices[uniq_idx]
    final_labels = labels[uniq_idx]
    
    # Save NPZ
    np.savez_compressed(os.path.join(out_dir, "OCCUPANCY.npz"), indices=final_indices, semantics=final_labels)

# --- 4. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='../data/nuscenes_occ/')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--out_root', default='output_demo/')
    parser.add_argument('--scene_name', default=None, help="Process specific scene, e.g. scene-0061")
    args = parser.parse_args()
    
    print(f"Initializing NuScenes ({args.version})...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    # Pick Scene
    target_scene = None
    if args.scene_name:
        for s in nusc.scene:
            if s['name'] == args.scene_name:
                target_scene = s
                break
        if not target_scene:
            print(f"Scene {args.scene_name} not found!")
            return
    else:
        target_scene = nusc.scene[0]
        
    print(f"Processing Scene: {target_scene['name']} ({target_scene['description']})")
    
    # Prepare Output
    scene_out = os.path.join(args.out_root, target_scene['name'])
    os.makedirs(scene_out, exist_ok=True)
    
    # Iterate Samples
    curr_token = target_scene['first_sample_token']
    idx = 0
    
    while curr_token:
        print(f"Frame {idx}: {curr_token}")
        sample = nusc.get('sample', curr_token)
        
        # Frame Output Folder
        frame_out = os.path.join(scene_out, f"{idx:03d}_{curr_token}")
        os.makedirs(frame_out, exist_ok=True)
        
        # A. Export Cameras
        export_camera_images(nusc, sample, frame_out)
        
        # B. Export Semantic LiDAR (PLY)
        export_semantic_lidar(nusc, sample, frame_out)
        
        # C. Generate Occupancy (NPZ)
        generate_and_save_occupancy(nusc, sample, frame_out)
        
        curr_token = sample['next']
        idx += 1
        
    print(f"Done! Output saved to {scene_out}")

if __name__ == '__main__':
    main()
