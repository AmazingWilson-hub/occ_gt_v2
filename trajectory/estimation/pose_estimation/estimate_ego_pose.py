#!/usr/bin/env python3
"""
Ego Pose Estimation from Raw Sensors (v2 - Fixed)
Compare 4 methods against NuScenes GT ego_pose:
  1. IMU Dead Reckoning
  2. LiDAR ICP
  3. GPS+IMU EKF Fusion
  4. Full Fusion (ICP + IMU + GPS)

Fixes from v1:
  - IMU rotation: apply cs_record to convert from sensor to ego frame
  - ICP: apply cs_record for LiDAR→Ego calibration
  - EKF: use CAN Pose orientation (already in ego frame) instead of raw IMU

Usage:
  python3 estimate_ego_pose.py [--scene scene-0061]
"""

import os
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import argparse

# ========================
# Data Loading
# ========================

DATAROOT = '/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ'

def load_nuscenes():
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version='v1.0-mini', dataroot=DATAROOT)

def load_canbus(scene_name, msg_type):
    """Load CAN bus JSON for a scene"""
    path = os.path.join(DATAROOT, 'can_bus', f'{scene_name}_{msg_type}.json')
    with open(path) as f:
        return json.load(f)

def get_scene_gt_poses(nusc, scene_name):
    """Get GT ego_pose for all keyframes in a scene"""
    scene = [s for s in nusc.scene if s['name'] == scene_name][0]
    
    poses = []
    sample_token = scene['first_sample_token']
    while sample_token:
        sample = nusc.get('sample', sample_token)
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego = nusc.get('ego_pose', sd['ego_pose_token'])
        
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([
            ego['rotation'][1], ego['rotation'][2],
            ego['rotation'][3], ego['rotation'][0]  # scipy: xyzw
        ]).as_matrix()
        T[:3, 3] = ego['translation']
        
        poses.append({
            'T': T,
            'utime': sd['timestamp'],
            'translation': np.array(ego['translation']),
            'rotation': np.array(ego['rotation']),  # wxyz
        })
        sample_token = sample['next'] if sample['next'] else None
    
    return poses

def get_scene_lidar_paths(nusc, scene_name):
    """Get LiDAR file paths for all keyframes"""
    scene = [s for s in nusc.scene if s['name'] == scene_name][0]
    
    paths = []
    sample_token = scene['first_sample_token']
    while sample_token:
        sample = nusc.get('sample', sample_token)
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = os.path.join(nusc.dataroot, sd['filename'])
        
        # Also get calibration
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        
        paths.append({
            'path': lidar_path,
            'utime': sd['timestamp'],
            'cs_record': cs,
        })
        sample_token = sample['next'] if sample['next'] else None
    
    return paths

def get_cs_transform(cs_record):
    """Build 4x4 transform from cs_record (LiDAR→Ego)"""
    T = np.eye(4)
    q = cs_record['rotation']  # wxyz
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = cs_record['translation']
    return T


# ========================
# Method 1: IMU Dead Reckoning (FIXED)
# ========================

def method_imu_deadreckoning(canbus_imu, canbus_pose, gt_poses):
    """
    Integrate IMU acceleration to get position.
    Use CAN Pose orientation (already in correct ego frame) for rotation.
    """
    utimes_gt = [p['utime'] for p in gt_poses]
    
    # Filter data to scene time range
    imu_data = [d for d in canbus_imu if d['utime'] >= utimes_gt[0] and d['utime'] <= utimes_gt[-1]]
    pose_data = [d for d in canbus_pose if d['utime'] >= utimes_gt[0] and d['utime'] <= utimes_gt[-1]]
    
    if len(imu_data) < 2:
        print("  WARNING: Not enough IMU data")
        return [np.eye(4)] * len(gt_poses)
    
    # Initialize with GT first pose
    pos = gt_poses[0]['translation'].copy()
    vel = np.zeros(3)
    g = np.array([0, 0, 9.81])  # gravity
    
    imu_trajectory = []
    
    for i in range(len(imu_data)):
        d = imu_data[i]
        
        # Use CAN Pose orientation (correct ego frame) instead of raw IMU q
        nearest_pose = min(pose_data, key=lambda p: abs(p['utime'] - d['utime']))
        q = nearest_pose['orientation']  # [w, x, y, z] already in ego frame
        R_ego = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        
        if i > 0:
            dt = (d['utime'] - imu_data[i-1]['utime']) / 1e6
            if dt <= 0 or dt > 0.1:
                continue
            
            # Acceleration in world frame (subtract gravity)
            accel_body = np.array(d['linear_accel'])
            accel_world = R_ego @ accel_body - g
            
            vel += accel_world * dt
            pos = pos + vel * dt
        
        T = np.eye(4)
        T[:3, :3] = R_ego
        T[:3, 3] = pos.copy()
        imu_trajectory.append((d['utime'], T.copy()))
    
    return interpolate_poses(imu_trajectory, utimes_gt)


# ========================
# Method 2: LiDAR ICP (FIXED)
# ========================

def method_lidar_icp(lidar_infos, canbus_imu, gt_poses, cs_record, voxel_size=0.5, max_range=80.0):
    """
    ICP registration between consecutive frames.
    FIX: Apply cs_record to convert ICP result from LiDAR frame to Ego frame.
    """
    n = len(lidar_infos)
    T_cs = get_cs_transform(cs_record)
    T_cs_inv = np.linalg.inv(T_cs)
    R_cs = T_cs[:3, :3]
    
    poses = [gt_poses[0]['T'].copy()]  # Start from GT first pose
    
    prev_down = load_and_preprocess_lidar(lidar_infos[0]['path'], voxel_size, max_range)
    cumulative = gt_poses[0]['T'].copy()
    
    for i in tqdm(range(1, n), desc="  ICP"):
        curr_down = load_and_preprocess_lidar(lidar_infos[i]['path'], voxel_size, max_range)
        
        # Initial guess from IMU (in sensor frame, relative rotation)
        init_T = np.eye(4)
        imu_near_prev = find_nearest_imu(canbus_imu, lidar_infos[i-1]['utime'])
        imu_near_curr = find_nearest_imu(canbus_imu, lidar_infos[i]['utime'])
        if imu_near_prev and imu_near_curr:
            R_prev = quat_to_rot(imu_near_prev['q'])
            R_curr = quat_to_rot(imu_near_curr['q'])
            init_T[:3, :3] = R_prev.T @ R_curr  # relative rotation in sensor frame
        
        # ICP in LiDAR frame
        reg = o3d.pipelines.registration.registration_icp(
            curr_down, prev_down,
            max_correspondence_distance=voxel_size * 3,
            init=init_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Convert ICP result from LiDAR frame to Ego frame:
        # T_ego = T_cs × T_icp × T_cs⁻¹
        T_icp_lidar = reg.transformation
        T_icp_ego = T_cs @ T_icp_lidar @ T_cs_inv
        
        cumulative = cumulative @ T_icp_ego
        poses.append(cumulative.copy())
        prev_down = curr_down
    
    return poses


def load_and_preprocess_lidar(path, voxel_size, max_range):
    """Load NuScenes LiDAR binary, preprocess for ICP"""
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    dists = np.linalg.norm(points, axis=1)
    pcd = pcd.select_by_index(np.where(dists < max_range)[0])
    
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return down


# ========================
# Method 3: GPS+IMU EKF (FIXED)
# ========================

def method_gps_imu_ekf(canbus_imu, canbus_pose, gt_poses, gps_hz=1.0):
    """
    Extended Kalman Filter fusing IMU (predict) + GPS (update).
    Use CAN Pose orientation (already in ego frame) for rotation.
    """
    utimes_gt = [p['utime'] for p in gt_poses]
    
    imu_data = [d for d in canbus_imu if utimes_gt[0] <= d['utime'] <= utimes_gt[-1]]
    pose_data = [d for d in canbus_pose if utimes_gt[0] <= d['utime'] <= utimes_gt[-1]]
    
    if len(imu_data) < 2 or len(pose_data) < 2:
        return [np.eye(4)] * len(gt_poses)
    
    # Subsample GPS to simulate real GPS frequency
    gps_interval = 1.0 / gps_hz * 1e6
    gps_measurements = []
    last_gps_time = 0
    for d in pose_data:
        if d['utime'] - last_gps_time >= gps_interval:
            gps_measurements.append(d)
            last_gps_time = d['utime']
    
    # EKF state: [px, py, pz, vx, vy, vz]
    x = np.zeros(6)
    x[:3] = gt_poses[0]['translation']
    P = np.eye(6) * 0.1
    
    Q_accel = 1.0
    R_gps_noise = np.eye(3) * 2.0
    g = np.array([0, 0, 9.81])
    
    trajectory = []
    gps_idx = 0
    
    for i in range(len(imu_data)):
        d = imu_data[i]
        # Use CAN Pose orientation (correct ego frame)
        nearest_pose = min(pose_data, key=lambda p: abs(p['utime'] - d['utime']))
        q = nearest_pose['orientation']
        R_ego = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        
        if i > 0:
            dt = (d['utime'] - imu_data[i-1]['utime']) / 1e6
            if dt <= 0 or dt > 0.1:
                continue
            
            # --- PREDICT (IMU) ---
            accel_body = np.array(d['linear_accel'])
            accel_world = R_ego @ accel_body - g
            
            F = np.eye(6)
            F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
            
            B = np.zeros((6, 3))
            B[3, 0] = dt; B[4, 1] = dt; B[5, 2] = dt
            B[0, 0] = 0.5 * dt**2; B[1, 1] = 0.5 * dt**2; B[2, 2] = 0.5 * dt**2
            
            x = F @ x + B @ accel_world
            
            Q = np.zeros((6, 6))
            Q[3:, 3:] = np.eye(3) * Q_accel * dt**2
            Q[:3, :3] = np.eye(3) * Q_accel * 0.25 * dt**4
            P = F @ P @ F.T + Q
            
            # --- UPDATE (GPS) ---
            while gps_idx < len(gps_measurements) and gps_measurements[gps_idx]['utime'] <= d['utime']:
                gps_pos = np.array(gps_measurements[gps_idx]['pos'])
                H = np.zeros((3, 6))
                H[:3, :3] = np.eye(3)
                
                y = gps_pos - H @ x
                S = H @ P @ H.T + R_gps_noise
                K = P @ H.T @ np.linalg.inv(S)
                
                x = x + K @ y
                P = (np.eye(6) - K @ H) @ P
                gps_idx += 1
        
        T = np.eye(4)
        T[:3, :3] = R_ego
        T[:3, 3] = x[:3]
        trajectory.append((d['utime'], T.copy()))
    
    return interpolate_poses(trajectory, utimes_gt)


# ========================
# Method 4: Full Fusion (FIXED)
# ========================

def method_full_fusion(lidar_infos, canbus_imu, canbus_pose, gt_poses, cs_record, voxel_size=0.5, max_range=80.0, gps_hz=1.0):
    """
    ICP for frame-to-frame, IMU for init guess, GPS for drift correction.
    FIX: ICP with cs_record, GPS correction with EKF-style update.
    """
    n = len(lidar_infos)
    utimes_gt = [p['utime'] for p in gt_poses]
    T_cs = get_cs_transform(cs_record)
    T_cs_inv = np.linalg.inv(T_cs)
    R_cs = T_cs[:3, :3]
    
    # Get GPS measurements (subsampled)
    pose_data = [d for d in canbus_pose if utimes_gt[0] <= d['utime'] <= utimes_gt[-1]]
    gps_interval = 1.0 / gps_hz * 1e6
    gps_list = []
    last_t = 0
    for d in pose_data:
        if d['utime'] - last_t >= gps_interval:
            gps_list.append({'utime': d['utime'], 'pos': np.array(d['pos'])})
            last_t = d['utime']
    
    poses = [gt_poses[0]['T'].copy()]
    cumulative = gt_poses[0]['T'].copy()
    prev_down = load_and_preprocess_lidar(lidar_infos[0]['path'], voxel_size, max_range)
    
    # EKF state for position smoothing
    x_pos = gt_poses[0]['translation'].copy()
    vel = np.zeros(3)
    P_pos = np.eye(3) * 0.1
    gps_idx = 0
    
    for i in tqdm(range(1, n), desc="  Full Fusion"):
        curr_down = load_and_preprocess_lidar(lidar_infos[i]['path'], voxel_size, max_range)
        
        # IMU initial guess (sensor frame, relative)
        init_T = np.eye(4)
        imu_prev = find_nearest_imu(canbus_imu, lidar_infos[i-1]['utime'])
        imu_curr = find_nearest_imu(canbus_imu, lidar_infos[i]['utime'])
        if imu_prev and imu_curr:
            R_prev = quat_to_rot(imu_prev['q'])
            R_curr = quat_to_rot(imu_curr['q'])
            init_T[:3, :3] = R_prev.T @ R_curr
        
        # ICP in LiDAR frame
        reg = o3d.pipelines.registration.registration_icp(
            curr_down, prev_down,
            max_correspondence_distance=voxel_size * 3,
            init=init_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # FIX: Convert ICP from LiDAR→Ego frame
        T_icp_ego = T_cs @ reg.transformation @ T_cs_inv
        cumulative = cumulative @ T_icp_ego
        
        # GPS correction: EKF-style update on position
        icp_pos = cumulative[:3, 3].copy()
        
        # Find nearest GPS within 0.5 sec
        curr_time = lidar_infos[i]['utime']
        nearest_gps = min(gps_list, key=lambda g: abs(g['utime'] - curr_time))
        if abs(nearest_gps['utime'] - curr_time) < 0.5e6:
            gps_pos = nearest_gps['pos']
            
            # EKF update for position
            R_gps_noise = np.eye(3) * 1.0  # GPS measurement noise
            R_icp_noise = np.eye(3) * 0.5  # ICP prediction noise (per step)
            
            P_pos = P_pos + R_icp_noise
            S = P_pos + R_gps_noise
            K = P_pos @ np.linalg.inv(S)
            
            innovation = gps_pos - icp_pos
            corrected_pos = icp_pos + K @ innovation
            P_pos = (np.eye(3) - K) @ P_pos
            
            cumulative[:3, 3] = corrected_pos
        
        poses.append(cumulative.copy())
        prev_down = curr_down
    
    return poses


# ========================
# Helper Functions
# ========================

def quat_to_rot(q_wxyz):
    """Convert wxyz quaternion to rotation matrix"""
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()

def find_nearest_imu(imu_data, utime):
    """Find IMU measurement nearest to given timestamp"""
    best = None
    best_dt = float('inf')
    for d in imu_data:
        dt = abs(d['utime'] - utime)
        if dt < best_dt:
            best_dt = dt
            best = d
        elif dt > best_dt:
            break
    return best

def interpolate_poses(trajectory, target_utimes):
    """Interpolate trajectory to target timestamps (nearest neighbor)"""
    result = []
    traj_times = [t[0] for t in trajectory]
    
    for ut in target_utimes:
        idx = np.argmin(np.abs(np.array(traj_times) - ut))
        result.append(trajectory[idx][1])
    
    return result


# ========================
# Evaluation
# ========================

def evaluate(estimated_poses, gt_poses, method_name):
    """Compute translation and rotation errors"""
    trans_errors = []
    rot_errors = []
    
    for est, gt in zip(estimated_poses, gt_poses):
        if isinstance(est, dict):
            est = est['T']
        gt_T = gt['T']
        
        te = np.linalg.norm(est[:3, 3] - gt_T[:3, 3])
        trans_errors.append(te)
        
        R_diff = est[:3, :3].T @ gt_T[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        rot_errors.append(np.degrees(angle))
    
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    
    print(f"\n{'='*50}")
    print(f"  {method_name}")
    print(f"{'='*50}")
    print(f"  Translation Error (m):")
    print(f"    Mean: {trans_errors.mean():.3f}")
    print(f"    Max:  {trans_errors.max():.3f}")
    print(f"    Std:  {trans_errors.std():.3f}")
    print(f"  Rotation Error (deg):")
    print(f"    Mean: {rot_errors.mean():.3f}")
    print(f"    Max:  {rot_errors.max():.3f}")
    print(f"    Std:  {rot_errors.std():.3f}")
    
    return {
        'name': method_name,
        'trans_mean': trans_errors.mean(),
        'trans_max': trans_errors.max(),
        'rot_mean': rot_errors.mean(),
        'rot_max': rot_errors.max(),
        'trans_errors': trans_errors,
        'rot_errors': rot_errors,
    }


def save_trajectory_comparison(all_results, gt_poses, out_dir):
    """Save trajectory comparison as CSV"""
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'trajectories.csv'), 'w') as f:
        methods = [r['name'] for r in all_results]
        header = 'frame,gt_x,gt_y,gt_z,' + ','.join(
            [f'{m}_x,{m}_y,{m}_z,{m}_te,{m}_re' for m in methods])
        f.write(header + '\n')
        
        for i, gt in enumerate(gt_poses):
            row = [str(i)]
            row.extend([f'{gt["T"][0,3]:.4f}', f'{gt["T"][1,3]:.4f}', f'{gt["T"][2,3]:.4f}'])
            for r in all_results:
                if isinstance(r['poses'][i], dict):
                    T = r['poses'][i]['T']
                else:
                    T = r['poses'][i]
                row.extend([
                    f'{T[0,3]:.4f}', f'{T[1,3]:.4f}', f'{T[2,3]:.4f}',
                    f'{r["trans_errors"][i]:.4f}', f'{r["rot_errors"][i]:.4f}'
                ])
            f.write(','.join(row) + '\n')
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':<25s} {'Trans(m)':>10s} {'Trans Max':>10s} {'Rot(deg)':>10s} {'Rot Max':>10s}")
    print(f"  {'-'*65}")
    for r in all_results:
        print(f"  {r['name']:<25s} {r['trans_mean']:>10.3f} {r['trans_max']:>10.3f} {r['rot_mean']:>10.3f} {r['rot_max']:>10.3f}")
    
    print(f"\n  Results saved to {out_dir}/trajectories.csv")


# ========================
# Main
# ========================

def main():
    parser = argparse.ArgumentParser(description="Ego Pose Estimation Comparison (v2 Fixed)")
    parser.add_argument('--scene', default='scene-0061', help="NuScenes scene name")
    parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(__file__), 'output'))
    args = parser.parse_args()
    
    print("Loading NuScenes...")
    nusc = load_nuscenes()
    
    scene_names = [s['name'] for s in nusc.scene]
    if args.scene not in scene_names:
        print(f"Scene {args.scene} not found. Available: {scene_names}")
        return
    
    print(f"\nScene: {args.scene}")
    
    # Load data
    print("Loading GT poses...")
    gt_poses = get_scene_gt_poses(nusc, args.scene)
    print(f"  {len(gt_poses)} keyframes")
    
    print("Loading CAN bus data...")
    canbus_imu = load_canbus(args.scene, 'ms_imu')
    canbus_pose = load_canbus(args.scene, 'pose')
    print(f"  IMU: {len(canbus_imu)} entries, Pose: {len(canbus_pose)} entries")
    
    print("Loading LiDAR paths...")
    lidar_infos = get_scene_lidar_paths(nusc, args.scene)
    cs_record = lidar_infos[0]['cs_record']  # calibration (same for all frames)
    print(f"  {len(lidar_infos)} frames")
    print(f"  cs_record translation: {cs_record['translation']}")
    
    all_results = []
    
    # Method 1: IMU Dead Reckoning
    print("\n--- Method 1: IMU Dead Reckoning ---")
    poses_imu = method_imu_deadreckoning(canbus_imu, canbus_pose, gt_poses)
    result = evaluate(poses_imu, gt_poses, "IMU Dead Reckoning")
    result['poses'] = poses_imu
    all_results.append(result)
    
    # Method 2: LiDAR ICP
    print("\n--- Method 2: LiDAR ICP ---")
    poses_icp = method_lidar_icp(lidar_infos, canbus_imu, gt_poses, cs_record)
    result = evaluate(poses_icp, gt_poses, "LiDAR ICP")
    result['poses'] = poses_icp
    all_results.append(result)
    
    # Method 3: GPS+IMU EKF
    print("\n--- Method 3: GPS+IMU EKF ---")
    poses_ekf = method_gps_imu_ekf(canbus_imu, canbus_pose, gt_poses)
    result = evaluate(poses_ekf, gt_poses, "GPS+IMU EKF")
    result['poses'] = poses_ekf
    all_results.append(result)
    
    # Method 4: Full Fusion
    print("\n--- Method 4: Full Fusion (ICP+IMU+GPS) ---")
    poses_full = method_full_fusion(lidar_infos, canbus_imu, canbus_pose, gt_poses, cs_record)
    result = evaluate(poses_full, gt_poses, "Full Fusion")
    result['poses'] = poses_full
    all_results.append(result)
    
    # Save results
    save_trajectory_comparison(all_results, gt_poses, args.out_dir)


if __name__ == "__main__":
    main()
