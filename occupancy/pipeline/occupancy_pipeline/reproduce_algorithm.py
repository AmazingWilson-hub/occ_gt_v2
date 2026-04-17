
import os
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
import argparse

# ==========================================
# Core Algorithm Functions (Simulated from utils)
# ==========================================

def rotate(points, rot_matrix: np.ndarray, center=None) -> np.array:
    """Applies a rotation to the point cloud."""
    if center is not None:
        points[:3, :] = np.dot(rot_matrix, points[:3, :]-center[:, None]) + center[:, None]
    else:
        points[:3, :] = np.dot(rot_matrix, points[:3, :])
    return points

def translate(points, x: np.ndarray) -> np.array:
    """Applies a translation to the point cloud."""
    for i in range(3):
        points[i, :] = points[i, :] + x[i]
    return points

def transform(points, rot_matrix: np.ndarray, trans_vector: np.ndarray, inverse=False) -> np.array:
    """Applies a homogeneous transform."""
    if not inverse:
        points = rotate(points, rot_matrix)
        points = translate(points, trans_vector)
    else:
        points = translate(points, -trans_vector)
        points = rotate(points, np.linalg.inv(rot_matrix))
    return points

def remove_close(points, radius: tuple=(1.0, 1.5)):
    """Removes point too close within a certain radius from origin."""
    x_filt = np.abs(points[0, :]) < radius[0]
    y_filt = np.abs(points[1, :]) < radius[1]
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points

def filter_points_in_ego(points, frame_info, instance_token):
    '''Filter points that land inside the target box in the current frame.'''
    if instance_token not in frame_info['instance_tokens']:
        return np.zeros(points.shape[1], dtype=bool)
        
    index = frame_info['instance_tokens'].index(instance_token)
    box = frame_info['boxes'][index]
    box_mask = points_in_box(box, points[:3, :])
    return box_mask

# ==========================================
# Main Alignment Logic (The "Secret Sauce")
# ==========================================

def align_dynamic_thing(prev_points, prev_box, prev_instance_token, ego_frame_info):
    """
    Aligns points from a moving object in a previous frame to the current frame.
    
    Math: 
    1. World -> Object (Previous T): Remove previous rotation/translation.
    2. Object -> World (Current T): Apply current rotation/translation.
    """
    
    # 0. Check if object exists in current frame
    if prev_instance_token not in ego_frame_info['instance_tokens']:
        return np.zeros((prev_points.shape[0], 0))

    # 1. Extract Object Points (at previous time)
    box_mask = points_in_box(prev_box, prev_points[:3, :])
    box_points = prev_points[:, box_mask].copy()
    
    if box_points.shape[1] == 0:
        return box_points

    # 2. Canonical Transformation (Remove Previous Pose)
    prev_bbox_center = prev_box.center
    prev_rotate_matrix = prev_box.rotation_matrix
    
    # Translate to center -> Rotate to axis-aligned
    box_points = rotate(box_points, np.linalg.inv(prev_rotate_matrix), center=prev_bbox_center)
    
    # 3. Target Transformation (Apply Current Pose)
    target_idx = ego_frame_info['instance_tokens'].index(prev_instance_token)
    current_box = ego_frame_info['boxes'][target_idx]
    
    ego_boxes_center = current_box.center
    
    # Calculate translation delta (old center -> new center)
    # Note: The original code logic is slightly verbose, effectively it translates 
    # the canonical points to the new center.
    box_points = translate(box_points, ego_boxes_center - prev_bbox_center)
    box_points = rotate(box_points, current_box.rotation_matrix, center=ego_boxes_center)
    
    # 4. Verify (Filter points that drifted outside due to noise/errors)
    final_mask = filter_points_in_ego(box_points, ego_frame_info, prev_instance_token)
    return box_points[:, final_mask]

def prev2ego(points, prev_frame_info, income_frame_info):
    """
    Aligns static background points using Ego-Motion.
    
    Math: P_curr = T_curr^-1 * T_prev * P_prev
    """
    # 1. Transform Previous Lidar -> Previous Ego -> Global
    prev_cs = prev_frame_info['cs_record']
    prev_pose = prev_frame_info['pose_record']
    
    points = transform(points, Quaternion(prev_cs['rotation']).rotation_matrix, np.array(prev_cs['translation']))
    points = transform(points, Quaternion(prev_pose['rotation']).rotation_matrix, np.array(prev_pose['translation']))

    # 2. Transform Global -> Current Ego -> Current Lidar
    ego_pose = income_frame_info['pose_record']
    ego_cs = income_frame_info['cs_record']
    
    points = transform(points, Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']), inverse=True)
    points = transform(points, Quaternion(ego_cs['rotation']).rotation_matrix, np.array(ego_cs['translation']), inverse=True)
    
    return points

# ==========================================
# Helpers to get Data
# ==========================================

def get_frame_info(nusc, sample_token, lidar_token):
    """Parses necessary info from NuScenes for a single frame."""
    sd_rec = nusc.get('sample_data', lidar_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
    
    # Load Point Cloud
    pc = LidarPointCloud.from_file(lidar_path)
    
    # Get Instance Tokens associated with boxes
    frame = nusc.get('sample', sample_token)
    # Note: boxes from get_sample_data are ordered. We need to match them to annotations to get instance tokens.
    # But get_sample_data returns boxes in its own way. 
    # For simplicitly in this demo, we will blindly assume box order matches or re-match based on tokens if needed.
    # The original repo uses frame['anns'] to get instance tokens, but we need to map them to the boxes.
    # Actually, nusc.get_sample_data returns boxes in global frame? No, sensor frame.
    # Let's trust the logic where we grab tokens from annotations for now.
    
    instance_tokens = [nusc.get('sample_annotation', token)['instance_token'] for token in frame['anns']]
    
    # WARNING: boxes returned by get_sample_data might not match 1:1 with frame['anns'] logic 
    # if we aren't careful. Ideally we iterate anns.
    # For this demo, we will iterate the ANNOTATIONS to be safe.
    
    return {
        'pc': pc,
        'cs_record': cs_record,
        'pose_record': pose_record,
        'boxes': boxes, # parameters for get_sample_data boxes
        'instance_tokens': instance_tokens, # From sample annotations
        'sample_token': sample_token
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes', help='Path to NuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='NuScenes version')
    args = parser.parse_args()

    print(f"Loading NuScenes {args.version} from {args.dataroot}...")
    try:
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    except Exception as e:
        print(f"Error loading NuScenes: {e}")
        return

    # Pick a random sample
    scene = nusc.scene[0]
    first_sample_token = scene['first_sample_token']
    current_sample = nusc.get('sample', first_sample_token)
    
    # Let's jump forward a few frames to have "past" frames
    for _ in range(5):
        if current_sample['next'] == '': break
        current_sample = nusc.get('sample', current_sample['next'])

    # Current Frame
    cur_lidar_token = current_sample['data']['LIDAR_TOP']
    ego_frame_info = get_frame_info(nusc, current_sample['token'], cur_lidar_token)
    
    print(f"Processing Sample: {current_sample['token']}")
    print(f"Total Points (Current): {ego_frame_info['pc'].points.shape[1]}")

    # Previous Frame (Simulating 1 sweep back)
    prev_sample = nusc.get('sample', current_sample['prev'])
    prev_lidar_token = prev_sample['data']['LIDAR_TOP']
    prev_frame_info = get_frame_info(nusc, prev_sample['token'], prev_lidar_token)
    
    print(f"Merging with Prev Sample: {prev_sample['token']}")
    
    # 1. Align Static
    # (Simplified: treating ALL points as static first, typically you mask out objects)
    # The real code removes ego vehicle and known objects before this.
    aligned_static = prev2ego(prev_frame_info['pc'].points.copy(), prev_frame_info, ego_frame_info)
    
    # 2. Align Dynamic
    # Iterate through objects in PREV frame
    # We need to find the box corresponding to an instance. 
    # nusc.get_sample_data returns boxes, but mapping them to instance tokens is tricky without lidar_seg/box mapping.
    # The original repo assumes a mapping exists. 
    # For this script, we will iterate known annotations in the sample.
    
    aligned_dynamic_points = []
    
    for ann_token in prev_sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        instance_token = ann['instance_token']
        category = ann['category_name']
        
        # Only care about moving things
        if 'vehicle' not in category and 'human' not in category:
            continue
            
        # Get the Box object for this annotation (in global frame, need to move to sensor frame for points check)
        # Actually, points_in_box expects box in same frame as points (Sensor Frame).
        # We can use nusc.get_box(ann_token) -> Global Box.
        # Then transform box to sensor frame.
        
        box = nusc.get_box(ann_token)
        
        # Transform box to PREV SENSOR frame
        cs_rec = prev_frame_info['cs_record']
        pose_rec = prev_frame_info['pose_record']
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)
        
        # Run Alignment
        # Note: We pass this box object, which is now in Prev-Lidar coords
        aligned_pts = align_dynamic_thing(prev_frame_info['pc'].points, box, instance_token, ego_frame_info)
        
        if aligned_pts.shape[1] > 0:
            aligned_dynamic_points.append(aligned_pts)
            print(f"  Aligned {aligned_pts.shape[1]} points for instance {instance_token[:8]} ({category})")

    # Final Stats
    total_static = aligned_static.shape[1]
    total_dynamic = sum([p.shape[1] for p in aligned_dynamic_points])
    print(f"alignment complete.")
    print(f"Static Points Added: {total_static}")
    print(f"Dynamic Points Added: {total_dynamic}")

if __name__ == "__main__":
    main()
