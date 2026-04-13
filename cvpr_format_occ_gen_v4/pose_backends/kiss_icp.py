"""
Backend: KISS-ICP (新方案)
用 kiss-icp 做純 LiDAR odometry，用 GT 第一幀做全域定位錨點，可選加 GPS EKF 修正漂移。

安裝：pip install kiss-icp
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# 加入 pose_estimation 路徑（為了取得 cs_record 和 GPS 修正工具）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pose_estimation'))


def _compute_timestamps(points):
    """
    為旋轉式 LiDAR 點雲生成 0~1 的 per-point timestamps。
    基於方位角（azimuth）模擬掃描順序，KISS-ICP 用於運動去畸變。
    """
    azimuth = np.arctan2(points[:, 1], points[:, 0])  # -π ~ π
    timestamps = (azimuth - azimuth.min()) / (azimuth.max() - azimuth.min() + 1e-10)
    return timestamps


def get_pose_dict(nusc, scene_name, use_gps_correction=True, voxel_size=0.5, max_range=80.0):
    """
    用 KISS-ICP 推算 ego pose，可選 GPS EKF 漂移修正。
    Returns: {utime: {'rotation': [w,x,y,z], 'translation': [x,y,z]}}
    """
    try:
        from kiss_icp.pipeline import KissICP
        from kiss_icp.config import KISSConfig
    except ImportError:
        raise ImportError(
            "kiss-icp 未安裝，請執行：pip install kiss-icp"
        )

    from estimate_ego_pose import (
        load_canbus, get_scene_gt_poses, get_scene_lidar_paths,
        get_cs_transform,
    )

    print("  [kiss_icp] Loading sensor data...")
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

    # — KISS-ICP 設定 —
    config = KISSConfig()
    config.data.max_range = max_range
    config.data.min_range = 0.5
    config.mapping.voxel_size = voxel_size

    kiss = KissICP(config)

    # 以 GT 第一幀作為全域坐標錨點
    T0_global = gt_poses[0]['T'].copy()
    cumulative = T0_global.copy()

    # EKF position state
    P_pos = np.eye(3) * 0.1
    R_icp_noise = np.eye(3) * 0.5
    R_gps_noise = np.eye(3) * 1.0

    pose_list = [cumulative.copy()]  # frame 0

    # 先處理第一幀（register 到 KISS-ICP 內部地圖）
    raw0 = np.fromfile(lidar_infos[0]['path'], dtype=np.float32).reshape(-1, 5)
    pts0 = raw0[:, :3]
    dists0 = np.linalg.norm(pts0, axis=1)
    mask0 = (dists0 > 0.5) & (dists0 < max_range)
    pts0 = pts0[mask0]
    ts0 = _compute_timestamps(pts0)
    kiss.register_frame(pts0, ts0)

    prev_kiss_pose = kiss.last_pose.copy()

    print(f"  [kiss_icp] Running on {len(lidar_infos)} frames...")
    for i in range(1, len(lidar_infos)):
        # 載入點雲（NuScenes .bin, float32, 5 channels）
        raw = np.fromfile(lidar_infos[i]['path'], dtype=np.float32).reshape(-1, 5)
        points = raw[:, :3]

        # 距離過濾
        dists = np.linalg.norm(points, axis=1)
        mask = (dists > 0.5) & (dists < max_range)
        points = points[mask]

        # 用方位角生成 0~1 的 per-point timestamps（旋轉式 LiDAR 掃描順序）
        timestamps = _compute_timestamps(points)

        # KISS-ICP 逐幀 register（sensor frame 內）
        kiss.register_frame(points, timestamps)

        # 取得增量 delta（sensor frame，from prev to curr）
        curr_kiss_pose = kiss.last_pose.copy()
        T_delta_sensor = np.linalg.inv(prev_kiss_pose) @ curr_kiss_pose
        prev_kiss_pose = curr_kiss_pose

        # 轉換到 Ego frame：T_ego = T_cs × T_delta × T_cs⁻¹
        T_delta_ego = T_cs @ T_delta_sensor @ T_cs_inv
        cumulative = cumulative @ T_delta_ego

        # GPS EKF 位置修正
        if gps_list:
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

        pose_list.append(cumulative.copy())

    # 組成 pose_dict
    pose_dict = {}
    for i, gt in enumerate(gt_poses):
        T = pose_list[i]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
        pose_dict[gt['utime']] = {
            'rotation': [q[3], q[0], q[1], q[2]],   # → wxyz
            'translation': T[:3, 3].tolist(),
        }

    suffix = "+GPS" if use_gps_correction else ""
    print(f"  [kiss_icp{suffix}] Done. {len(pose_dict)} poses estimated.")
    return pose_dict
