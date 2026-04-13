"""
Backend: VGICP Fusion (small_gicp VGICP + IMU init + GPS EKF)
用 small_gicp 的 VGICP (Voxelized Generalized ICP) 替代 Open3D Point-to-Plane ICP，
理論上配準精度更高（用點雲協方差矩陣而非僅法向量）。
其餘架構與 Full Fusion 相同：IMU 提供初始旋轉猜測，GPS EKF 修正位置漂移。

安裝：pip install small_gicp
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pose_estimation'))


def get_pose_dict(nusc, scene_name, voxel_size=0.5, max_range=80.0, gps_hz=1.0):
    """
    用 VGICP + IMU init + GPS EKF 推算 ego pose。
    Returns: {utime: {'rotation': [w,x,y,z], 'translation': [x,y,z]}}
    """
    try:
        import small_gicp
    except ImportError:
        raise ImportError("small_gicp 未安裝，請執行：pip install small_gicp")

    from estimate_ego_pose import (
        load_canbus, get_scene_gt_poses, get_scene_lidar_paths,
        get_cs_transform, find_nearest_imu, quat_to_rot,
    )

    print("  [vgicp_fusion] Loading sensor data...")
    gt_poses = get_scene_gt_poses(nusc, scene_name)
    canbus_imu = load_canbus(scene_name, 'ms_imu')
    canbus_pose = load_canbus(scene_name, 'pose')
    lidar_infos = get_scene_lidar_paths(nusc, scene_name)
    cs_record = lidar_infos[0]['cs_record']
    T_cs = get_cs_transform(cs_record)
    T_cs_inv = np.linalg.inv(T_cs)

    n = len(lidar_infos)
    utimes_gt = [p['utime'] for p in gt_poses]

    # GPS（子取樣）
    pose_data = [d for d in canbus_pose if utimes_gt[0] <= d['utime'] <= utimes_gt[-1]]
    gps_interval = 1.0 / gps_hz * 1e6
    gps_list = []
    last_t = 0
    for d in pose_data:
        if d['utime'] - last_t >= gps_interval:
            gps_list.append({'utime': d['utime'], 'pos': np.array(d['pos'])})
            last_t = d['utime']

    # 初始化
    poses = [gt_poses[0]['T'].copy()]
    cumulative = gt_poses[0]['T'].copy()

    # EKF position state
    P_pos = np.eye(3) * 0.1
    R_icp_noise = np.eye(3) * 0.3   # VGICP 噪聲比 Point-to-Plane 低
    R_gps_noise = np.eye(3) * 1.0

    # 預載入第一幀點雲
    prev_pts = _load_points(lidar_infos[0]['path'], max_range).astype(np.float64)

    print(f"  [vgicp_fusion] Running VGICP on {n} frames...")
    for i in tqdm(range(1, n), desc="  VGICP Fusion"):
        curr_pts = _load_points(lidar_infos[i]['path'], max_range).astype(np.float64)

        # IMU 初始猜測（sensor frame，相對旋轉）
        init_T = np.eye(4)
        imu_prev = find_nearest_imu(canbus_imu, lidar_infos[i-1]['utime'])
        imu_curr = find_nearest_imu(canbus_imu, lidar_infos[i]['utime'])
        if imu_prev and imu_curr:
            R_prev = quat_to_rot(imu_prev['q'])
            R_curr = quat_to_rot(imu_curr['q'])
            init_T[:3, :3] = R_prev.T @ R_curr

        # VGICP 配準（raw numpy overload，支援 registration_type='VGICP'）
        result = small_gicp.align(
            prev_pts, curr_pts,
            init_T_target_source=init_T,
            registration_type='VGICP',
            voxel_resolution=voxel_size,
            downsampling_resolution=voxel_size,
            max_correspondence_distance=voxel_size * 3,
            num_threads=4,
        )

        # result.T_target_source：把 source（curr）對齊到 target（prev）的變換
        T_vgicp_lidar = result.T_target_source

        # 轉到 Ego frame：T_ego = T_cs × T_vgicp × T_cs⁻¹
        T_vgicp_ego = T_cs @ T_vgicp_lidar @ T_cs_inv
        cumulative = cumulative @ T_vgicp_ego

        # GPS EKF 位置修正
        curr_time = lidar_infos[i]['utime']
        nearest_gps = min(gps_list, key=lambda g: abs(g['utime'] - curr_time))
        if abs(nearest_gps['utime'] - curr_time) < 0.5e6:
            icp_pos = cumulative[:3, 3].copy()
            gps_pos = nearest_gps['pos']

            P_pos = P_pos + R_icp_noise
            S = P_pos + R_gps_noise
            K = P_pos @ np.linalg.inv(S)
            innovation = gps_pos - icp_pos
            corrected_pos = icp_pos + K @ innovation
            P_pos = (np.eye(3) - K) @ P_pos
            cumulative[:3, 3] = corrected_pos

        poses.append(cumulative.copy())
        prev_pts = curr_pts

    # 組成 pose_dict
    pose_dict = {}
    for i, gt in enumerate(gt_poses):
        T = poses[i]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
        pose_dict[gt['utime']] = {
            'rotation': [q[3], q[0], q[1], q[2]],
            'translation': T[:3, 3].tolist(),
        }

    print(f"  [vgicp_fusion] Done. {len(pose_dict)} poses estimated.")
    return pose_dict


def _load_points(path, max_range):
    """載入 NuScenes LiDAR .bin 並做距離過濾"""
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    points = raw[:, :3]
    dists = np.linalg.norm(points, axis=1)
    mask = (dists > 0.5) & (dists < max_range)
    return points[mask]
