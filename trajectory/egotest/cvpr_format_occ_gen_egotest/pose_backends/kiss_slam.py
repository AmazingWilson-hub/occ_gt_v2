"""
Backend: KISS-SLAM (新方案 — 含 Loop Closure)
用 kiss-slam 做 LiDAR SLAM，包含 Loop Closure + Pose Graph Optimization。
理論上比 KISS-ICP 更好，因為能修正累積漂移。

安裝：pip install kiss-slam
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'estimation', 'pose_estimation'))


def _compute_timestamps(points):
    """
    為旋轉式 LiDAR 點雲生成 0~1 的 per-point timestamps。
    基於方位角（azimuth）模擬掃描順序。
    """
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    timestamps = (azimuth - azimuth.min()) / (azimuth.max() - azimuth.min() + 1e-10)
    return timestamps


def get_pose_dict(nusc, scene_name, use_gps_correction=True, max_range=80.0):
    """
    用 KISS-SLAM（含 Loop Closure + Pose Graph Optimization）推算 ego pose。
    Returns: {utime: {'rotation': [w,x,y,z], 'translation': [x,y,z]}}
    """
    try:
        from kiss_slam.slam import KissSLAM
        from kiss_slam.config import KissSLAMConfig
    except ImportError:
        raise ImportError(
            "kiss-slam 未安裝，請執行：pip install kiss-slam"
        )

    from estimate_ego_pose import (
        load_canbus, get_scene_gt_poses, get_scene_lidar_paths,
        get_cs_transform,
    )

    print("  [kiss_slam] Loading sensor data...")
    gt_poses = get_scene_gt_poses(nusc, scene_name)
    lidar_infos = get_scene_lidar_paths(nusc, scene_name)
    cs_record = lidar_infos[0]['cs_record']
    T_cs = get_cs_transform(cs_record)
    T_cs_inv = np.linalg.inv(T_cs)

    # GPS 修正用
    if use_gps_correction:
        canbus_pose = load_canbus(scene_name, 'pose')
        utimes_gt = [p['utime'] for p in gt_poses]
        pose_data = [d for d in canbus_pose if utimes_gt[0] <= d['utime'] <= utimes_gt[-1]]
        gps_list = [{'utime': d['utime'], 'pos': np.array(d['pos'])} for d in pose_data]
    else:
        gps_list = []

    # — KISS-SLAM 設定 —
    config = KissSLAMConfig()
    config.odometry.preprocessing.max_range = max_range
    config.odometry.preprocessing.min_range = 0.5
    config.odometry.mapping.voxel_size = max_range / 100.0  # 0.8 for 80m
    config.local_mapper.voxel_size = 0.5
    # 對於短序列（NuScenes mini ~39 frames），用較小的 splitting distance
    # 讓 SLAM 有機會建立更多 local map nodes 做 loop closure
    config.local_mapper.splitting_distance = 30.0

    kiss_slam = KissSLAM(config)

    # 逐幀處理
    print(f"  [kiss_slam] Processing {len(lidar_infos)} frames...")
    for i in range(len(lidar_infos)):
        raw = np.fromfile(lidar_infos[i]['path'], dtype=np.float32).reshape(-1, 5)
        points = raw[:, :3]

        dists = np.linalg.norm(points, axis=1)
        mask = (dists > 0.5) & (dists < max_range)
        points = points[mask]

        timestamps = _compute_timestamps(points)
        kiss_slam.process_scan(points, timestamps)

    # 最後一個 local map node 收尾
    kiss_slam.generate_new_node()
    kiss_slam.local_map_graph.erase_last_local_map()

    # 全局 Pose Graph Optimization（含 Loop Closure 修正）
    print(f"  [kiss_slam] Running Pose Graph Optimization...")
    print(f"  [kiss_slam] Found {len(kiss_slam.closures)} loop closures")
    optimized_poses, _ = kiss_slam.fine_grained_optimization()
    optimized_poses = np.array(optimized_poses)

    n_frames = len(lidar_infos)
    if len(optimized_poses) < n_frames:
        print(f"  [kiss_slam WARNING] Got {len(optimized_poses)} poses for {n_frames} frames, padding last")
        while len(optimized_poses) < n_frames:
            optimized_poses = np.concatenate([optimized_poses, optimized_poses[-1:]], axis=0)
    elif len(optimized_poses) > n_frames:
        optimized_poses = optimized_poses[:n_frames]

    # KISS-SLAM 的 pose 在 sensor frame，需要轉到 ego frame 再定位到全域
    # 用 GT 第一幀做全域錨點
    T0_global = gt_poses[0]['T'].copy()
    T0_sensor = optimized_poses[0]
    T0_sensor_inv = np.linalg.inv(T0_sensor)

    pose_list = []
    # EKF position state
    P_pos = np.eye(3) * 0.1
    R_slam_noise = np.eye(3) * 0.3   # SLAM 比純 ICP 噪聲更低
    R_gps_noise = np.eye(3) * 1.0

    for i in range(n_frames):
        # 相對於第一幀的增量（sensor frame）
        T_rel_sensor = T0_sensor_inv @ optimized_poses[i]

        # 轉到 ego frame
        T_rel_ego = T_cs @ T_rel_sensor @ T_cs_inv

        # 全域定位
        T_global = T0_global @ T_rel_ego

        # GPS EKF 修正
        if gps_list and i > 0:
            curr_time = lidar_infos[i]['utime']
            nearest_gps = min(gps_list, key=lambda g: abs(g['utime'] - curr_time))
            if abs(nearest_gps['utime'] - curr_time) < 0.5e6:
                slam_pos = T_global[:3, 3].copy()
                gps_pos = nearest_gps['pos']

                P_pos = P_pos + R_slam_noise
                S = P_pos + R_gps_noise
                K = P_pos @ np.linalg.inv(S)
                innovation = gps_pos - slam_pos
                corrected_pos = slam_pos + K @ innovation
                P_pos = (np.eye(3) - K) @ P_pos
                T_global[:3, 3] = corrected_pos

        pose_list.append(T_global.copy())

    # 組成 pose_dict
    pose_dict = {}
    for i, gt in enumerate(gt_poses):
        T = pose_list[i]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
        pose_dict[gt['utime']] = {
            'rotation': [q[3], q[0], q[1], q[2]],
            'translation': T[:3, 3].tolist(),
        }

    suffix = "+GPS" if use_gps_correction else ""
    print(f"  [kiss_slam{suffix}] Done. {len(pose_dict)} poses, {len(kiss_slam.closures)} closures.")
    return pose_dict
