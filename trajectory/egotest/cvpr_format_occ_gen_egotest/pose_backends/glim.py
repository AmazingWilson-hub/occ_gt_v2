"""
Backend: GLIM-Style (VGICP + Pose Graph Optimization + Loop Closure)
復刻 koide3/glim 的核心演算法思路：
  - small_gicp VGICP 做 frame-to-frame scan matching
  - IMU 提供旋轉初始猜測
  - Distance-based Loop Closure 偵測 + VGICP 驗證
  - Pose Graph Optimization（Levenberg-Marquardt）全域修正軌跡
  - GPS EKF 位置修正

與 vgicp_fusion 相比，多了 Loop Closure + Pose Graph Optimization。
與 kiss_slam 相比，用 VGICP 取代 KISS-ICP，且加入 IMU 初始猜測。

安裝：pip install small_gicp
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'estimation', 'pose_estimation'))


# ───── Pose Graph Optimization ─────

def _pose_to_vec(T):
    """SE(3) 4x4 → 6-vector [tx, ty, tz, rx, ry, rz] (rotation vector)"""
    t = T[:3, 3]
    r = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([t, r])


def _vec_to_pose(v):
    """6-vector → SE(3) 4x4"""
    T = np.eye(4)
    T[:3, 3] = v[:3]
    T[:3, :3] = Rotation.from_rotvec(v[3:6]).as_matrix()
    return T


def _relative_pose(T_i, T_j):
    """計算 T_i 到 T_j 的相對變換"""
    return np.linalg.inv(T_i) @ T_j


def _pose_graph_residuals(x, n_poses, edges):
    """
    Pose graph residuals for least_squares optimization.
    x: flattened [6*n_poses] vector of all poses
    edges: list of (i, j, T_ij_measured, weight)
    """
    poses = x.reshape(n_poses, 6)
    residuals = []

    for (i, j, T_ij_meas, weight) in edges:
        T_i = _vec_to_pose(poses[i])
        T_j = _vec_to_pose(poses[j])
        T_ij_est = _relative_pose(T_i, T_j)
        # Residual = log(T_ij_meas^{-1} @ T_ij_est) as 6-vector
        T_err = np.linalg.inv(T_ij_meas) @ T_ij_est
        err_vec = _pose_to_vec(T_err)  # should be ~0 if perfect
        residuals.extend(err_vec * weight)

    return np.array(residuals)


def _optimize_pose_graph(initial_poses, edges):
    """
    Pose Graph Optimization using scipy Levenberg-Marquardt.
    initial_poses: list of SE(3) 4x4
    edges: list of (i, j, T_ij_measured, weight)
    Returns: list of optimized SE(3) 4x4
    """
    n = len(initial_poses)
    x0 = np.array([_pose_to_vec(T) for T in initial_poses]).flatten()

    # Fix first pose (anchor)
    def residuals_fn(x_free):
        x_full = np.concatenate([_pose_to_vec(initial_poses[0]), x_free])
        return _pose_graph_residuals(x_full, n, edges)

    x_free0 = x0[6:]  # exclude first pose
    result = least_squares(residuals_fn, x_free0, method='lm', max_nfev=200)

    # Reconstruct full pose list
    x_opt = np.concatenate([_pose_to_vec(initial_poses[0]), result.x])
    poses_opt = x_opt.reshape(n, 6)
    return [_vec_to_pose(v) for v in poses_opt]


# ───── Loop Closure Detection ─────

def _check_registration_quality(T_odom, T_vgicp, max_trans_err=2.0, max_rot_deg=10.0):
    """
    檢查 VGICP 結果是否與 odometry 預測一致。
    如果 VGICP 結果偏差過大，可能是錯誤配準。
    """
    # 位移差異
    trans_err = np.linalg.norm(T_odom[:3, 3] - T_vgicp[:3, 3])

    # 旋轉差異 (degrees)
    R_err = T_odom[:3, :3].T @ T_vgicp[:3, :3]
    angle_err = np.abs(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
    angle_err_deg = np.degrees(angle_err)

    return trans_err < max_trans_err and angle_err_deg < max_rot_deg, trans_err, angle_err_deg


def _detect_loop_closures(poses_sensor, points_list, T_cs,
                          dist_thresh=15.0, time_gap=8,
                          voxel_size=0.5):
    """
    Distance-based loop closure detection + VGICP 驗證 + 品質過濾。
    poses_sensor: list of SE(3) 4x4 in sensor frame (odometry estimates)
    points_list: list of downsampled point clouds
    dist_thresh: 空間距離閾值 (m)
    time_gap: 最小時間間隔（幀數），避免相鄰幀誤判
    Returns: list of (i, j, T_ij_vgicp)
    """
    import small_gicp

    n = len(poses_sensor)
    closures = []
    rejected = 0

    # 計算全域位置（用 sensor frame poses）
    positions = np.array([T[:3, 3] for T in poses_sensor])

    for i in range(n):
        for j in range(i + time_gap, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < dist_thresh:
                # VGICP 驗證
                T_init = _relative_pose(poses_sensor[i], poses_sensor[j])
                try:
                    result = small_gicp.align(
                        points_list[i].astype(np.float64),
                        points_list[j].astype(np.float64),
                        init_T_target_source=T_init,
                        registration_type='VGICP',
                        voxel_resolution=voxel_size,
                        downsampling_resolution=voxel_size,
                        max_correspondence_distance=voxel_size * 3,
                        num_threads=4,
                    )
                    T_ij = result.T_target_source

                    # 品質檢查：VGICP 結果不應與 odometry 預測偏差太大
                    ok, t_err, r_err = _check_registration_quality(T_init, T_ij)
                    if ok:
                        closures.append((i, j, T_ij))
                    else:
                        rejected += 1
                except Exception:
                    rejected += 1

    if rejected:
        print(f"  [glim] Loop closure: {rejected} candidates rejected (quality check)")

    return closures


# ───── Main Backend ─────

def get_pose_dict(nusc, scene_name, voxel_size=0.5, max_range=80.0,
                  loop_dist_thresh=15.0, loop_time_gap=8):
    """
    用 GLIM-Style 方法推算 ego pose：
      1. VGICP odometry (frame-to-frame) + IMU init
      2. Loop Closure detection + VGICP validation
      3. Pose Graph Optimization
      4. GPS EKF position correction
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

    print("  [glim] Loading sensor data...")
    gt_poses = get_scene_gt_poses(nusc, scene_name)
    canbus_imu = load_canbus(scene_name, 'ms_imu')
    canbus_pose = load_canbus(scene_name, 'pose')
    lidar_infos = get_scene_lidar_paths(nusc, scene_name)
    cs_record = lidar_infos[0]['cs_record']
    T_cs = get_cs_transform(cs_record)
    T_cs_inv = np.linalg.inv(T_cs)

    n = len(lidar_infos)
    utimes_gt = [p['utime'] for p in gt_poses]

    # GPS data
    pose_data = [d for d in canbus_pose if utimes_gt[0] <= d['utime'] <= utimes_gt[-1]]
    gps_list = [{'utime': d['utime'], 'pos': np.array(d['pos'])} for d in pose_data]

    # ── Step 1: VGICP Odometry + IMU init ──
    print(f"  [glim] Step 1/3: VGICP Odometry on {n} frames...")
    odom_poses_sensor = [np.eye(4)]  # sensor frame, first frame is identity
    odom_edges = []                  # (i, j, T_ij_measured, weight)
    points_cache = []                # for loop closure

    prev_pts = _load_points(lidar_infos[0]['path'], max_range).astype(np.float64)
    points_cache.append(prev_pts)

    for i in tqdm(range(1, n), desc="  GLIM VGICP Odom"):
        curr_pts = _load_points(lidar_infos[i]['path'], max_range).astype(np.float64)
        points_cache.append(curr_pts)

        # IMU 初始猜測（sensor frame 相對旋轉）
        init_T = np.eye(4)
        imu_prev = find_nearest_imu(canbus_imu, lidar_infos[i-1]['utime'])
        imu_curr = find_nearest_imu(canbus_imu, lidar_infos[i]['utime'])
        if imu_prev and imu_curr:
            R_prev = quat_to_rot(imu_prev['q'])
            R_curr = quat_to_rot(imu_curr['q'])
            init_T[:3, :3] = R_prev.T @ R_curr

        # VGICP 配準
        result = small_gicp.align(
            prev_pts, curr_pts,
            init_T_target_source=init_T,
            registration_type='VGICP',
            voxel_resolution=voxel_size,
            downsampling_resolution=voxel_size,
            max_correspondence_distance=voxel_size * 3,
            num_threads=4,
        )

        T_delta_sensor = result.T_target_source
        odom_pose = odom_poses_sensor[-1] @ T_delta_sensor
        odom_poses_sensor.append(odom_pose)

        # Odometry edge (weight=1.0 for sequential)
        odom_edges.append((i-1, i, T_delta_sensor, 1.0))
        prev_pts = curr_pts

    # ── Step 2: Loop Closure Detection ──
    print(f"  [glim] Step 2/3: Loop Closure Detection...")
    closures = _detect_loop_closures(
        odom_poses_sensor, points_cache, T_cs,
        dist_thresh=loop_dist_thresh,
        time_gap=loop_time_gap,
        voxel_size=voxel_size,
    )
    print(f"  [glim] Found {len(closures)} loop closures")

    # Add loop closure edges (weight=0.3, less trusted than odom to avoid over-correction)
    for (i, j, T_ij) in closures:
        odom_edges.append((i, j, T_ij, 0.3))

    # ── Step 3: Pose Graph Optimization ──
    if closures:
        print(f"  [glim] Step 3/3: Pose Graph Optimization ({len(odom_edges)} edges)...")
        optimized_sensor = _optimize_pose_graph(odom_poses_sensor, odom_edges)
    else:
        print(f"  [glim] Step 3/3: No loop closures, skipping PGO")
        optimized_sensor = odom_poses_sensor

    # ── Convert to global frame + GPS EKF ──
    T0_global = gt_poses[0]['T'].copy()
    T0_sensor_inv = np.linalg.inv(optimized_sensor[0])

    P_pos = np.eye(3) * 0.1
    R_vgicp_noise = np.eye(3) * 0.3
    R_gps_noise = np.eye(3) * 1.0

    pose_list = []
    for i in range(n):
        # Relative to frame 0 in sensor coords
        T_rel_sensor = T0_sensor_inv @ optimized_sensor[i]

        # Sensor → Ego frame
        T_rel_ego = T_cs @ T_rel_sensor @ T_cs_inv

        # Global positioning
        T_global = T0_global @ T_rel_ego

        # GPS EKF correction
        if gps_list and i > 0:
            curr_time = lidar_infos[i]['utime']
            nearest_gps = min(gps_list, key=lambda g: abs(g['utime'] - curr_time))
            if abs(nearest_gps['utime'] - curr_time) < 0.5e6:
                est_pos = T_global[:3, 3].copy()
                gps_pos = nearest_gps['pos']

                P_pos = P_pos + R_vgicp_noise
                S = P_pos + R_gps_noise
                K = P_pos @ np.linalg.inv(S)
                innovation = gps_pos - est_pos
                corrected_pos = est_pos + K @ innovation
                P_pos = (np.eye(3) - K) @ P_pos
                T_global[:3, 3] = corrected_pos

        pose_list.append(T_global.copy())

    # Build pose_dict
    pose_dict = {}
    for i, gt in enumerate(gt_poses):
        T = pose_list[i]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
        pose_dict[gt['utime']] = {
            'rotation': [q[3], q[0], q[1], q[2]],  # → wxyz
            'translation': T[:3, 3].tolist(),
        }

    print(f"  [glim] Done. {len(pose_dict)} poses, {len(closures)} loop closures.")
    return pose_dict


def _load_points(path, max_range):
    """載入 NuScenes LiDAR .bin 並做距離過濾"""
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    points = raw[:, :3]
    dists = np.linalg.norm(points, axis=1)
    mask = (dists > 0.5) & (dists < max_range)
    return points[mask]
