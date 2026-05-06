#!/usr/bin/env python3
"""
NuScenes lane line pipeline:
  1. Per-frame: RANSAC ground plane + KMeans intensity → lane marking points
  2. Accumulate in world frame across all samples in a scene
  3. Fit polynomial lane lines
  4. Project fitted lanes onto CAM_FRONT and write video

Usage:
    python3 lane_line/nusc_lane_pipeline.py \
        --scene_idx 0 \
        --version v1.0-mini \
        --out video_out/nusc_lane.mp4
"""

import os, sys, json, argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

REPO     = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATAROOT = '/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ'

LANE_COLORS = [
    (  0, 255,   0),
    (  0, 255, 255),
    (255,   0,   0),
    (  0, 128, 255),
    (255,   0, 255),
    (255, 128,   0),
    (128,   0, 255),
    (  0, 200, 128),
]

# ── Coordinate helpers ────────────────────────────────────────────────────────

def quat_to_rot(q_wxyz):
    """[w,x,y,z] → 3x3 rotation matrix."""
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()

def make_T(rot_wxyz, trans_xyz):
    """Build 4x4 SE3 transform from quaternion (wxyz) and translation."""
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(rot_wxyz)
    T[:3,  3] = trans_xyz
    return T

def apply_T(T, pts):
    """Nx3 → Nx3 via 4x4 transform."""
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


# ── Lane filtering (RANSAC + KMeans intensity) ────────────────────────────────

def filter_lane_pts(pts_xyz, intensity, roi_fwd=(2, 60), roi_lat=(-10, 10)):
    """Return Nx3 lane marking points in LiDAR/ego frame."""
    from sklearn import linear_model
    from sklearn.cluster import KMeans

    # ROI crop
    mask = ((pts_xyz[:, 0] > roi_fwd[0]) & (pts_xyz[:, 0] < roi_fwd[1]) &
            (pts_xyz[:, 1] > roi_lat[0]) & (pts_xyz[:, 1] < roi_lat[1]))
    pts  = pts_xyz[mask]
    inty = intensity[mask]
    if len(pts) < 20:
        return np.zeros((0, 3))

    # RANSAC ground plane  Z = aX + bY + c
    try:
        ransac = linear_model.RANSACRegressor(
            linear_model.LinearRegression(),
            min_samples=5, residual_threshold=0.2, max_trials=150)
        ransac.fit(pts[:, :2], pts[:, 2])
        ground = ransac.inlier_mask_
    except Exception:
        ground = pts[:, 2] < -1.0

    pts_g  = pts[ground]
    inty_g = inty[ground]
    if len(pts_g) < 10:
        return np.zeros((0, 3))

    # KMeans intensity clustering — remove largest cluster (asphalt)
    inty_norm = inty_g / 255.0
    i_range   = inty_norm.max() - inty_norm.min()
    if i_range < 0.1:
        return np.zeros((0, 3))

    n_clusters = 2 if i_range > 0.4 else 3
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(inty_norm.reshape(-1, 1))
    largest = np.argmax(np.bincount(labels))
    keep = labels != largest

    return pts_g[keep]


# ── Lane fitting (same logic as fit_lanes.py) ─────────────────────────────────

def cluster_by_lateral(pts, min_pts=30, bw=0.25):
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    ys = pts[:, 1]
    y_min, y_max = ys.min() - 1.0, ys.max() + 1.0
    y_grid = np.linspace(y_min, y_max, 2000)
    dy = y_grid[1] - y_grid[0]
    kde = gaussian_kde(ys, bw_method=bw / ys.std())
    density = kde(y_grid)
    peaks, _ = find_peaks(density, distance=1.0 / dy, prominence=density.max() * 0.01)
    if len(peaks) == 0:
        return [pts] if len(pts) >= min_pts else []

    cut_ys = []
    for i in range(len(peaks) - 1):
        lo, hi = peaks[i], peaks[i + 1]
        valley_idx = lo + np.argmin(density[lo:hi + 1])
        cut_ys.append(y_grid[valley_idx])

    order = np.argsort(ys)
    split_pos = np.searchsorted(ys[order], cut_ys)
    groups = np.split(order, split_pos)
    return [pts[g] for g in groups if len(g) >= min_pts]


def fit_lane(pts, degree=3, n_samples=300, smooth_window=21):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x_range = (x.min(), x.max())
    if x_range[1] - x_range[0] < 5.0:
        return None
    coeff_y = np.polyfit(x, y, degree)
    coeff_z = np.polyfit(x, z, degree)
    x_clean = np.linspace(x_range[0], x_range[1], n_samples)
    y_clean = np.polyval(coeff_y, x_clean)
    z_clean = np.polyval(coeff_z, x_clean)
    if smooth_window and n_samples > smooth_window:
        y_clean = savgol_filter(y_clean, smooth_window, 2)
        z_clean = savgol_filter(z_clean, smooth_window, 2)
    return np.stack([x_clean, y_clean, z_clean], axis=1)


# ── Camera projection ─────────────────────────────────────────────────────────

def project_and_draw_lane(img, pts_world, T_world_ego_inv, T_cam_ego_inv,
                          K, W, H, color, thickness=3):
    """world → ego → camera → image pixels, draw lane."""
    pts_ego = apply_T(T_world_ego_inv, pts_world)

    # Ego-frame range filter (no extrapolation hooks)
    in_range = ((pts_ego[:, 0] > -5)  & (pts_ego[:, 0] < 80) &
                (np.abs(pts_ego[:, 1]) < 15))
    pts_ego = pts_ego[in_range]
    if len(pts_ego) < 2:
        return

    pts_cam = apply_T(T_cam_ego_inv, pts_ego)

    # Angle filter: NuScenes cameras have no distortion coeff,
    # but still skip very wide-angle points
    z = pts_cam[:, 2]
    tx = np.abs(pts_cam[:, 0]) / np.maximum(z, 1e-6)
    ty = np.abs(pts_cam[:, 1]) / np.maximum(z, 1e-6)
    front = (z > 0.1) & (tx < 2.0) & (ty < 2.0)
    if not front.any():
        return

    # Project (pinhole, no distortion for NuScenes)
    p = pts_cam[front]
    u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2])
    v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2])

    uvs = np.full((len(pts_ego), 2), -1.0)
    uvs[front] = np.stack([u, v], axis=1)

    in_b = np.zeros(len(pts_ego), bool)
    in_b[front] = ((u >= 0) & (u < W) & (v >= 0) & (v < H))

    for i in range(len(pts_ego) - 1):
        if in_b[i] and in_b[i + 1]:
            cv2.line(img,
                     tuple(uvs[i].astype(np.int32)),
                     tuple(uvs[i + 1].astype(np.int32)),
                     color, thickness, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_idx', type=int, default=0,
                        help='Index into nusc.scene list')
    parser.add_argument('--version',   default='v1.0-mini')
    parser.add_argument('--cam',       default='CAM_FRONT')
    parser.add_argument('--out',       default=None)
    parser.add_argument('--fps',       type=int, default=2)
    parser.add_argument('--degree',    type=int, default=3)
    parser.add_argument('--min_pts',   type=int, default=30)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud

    nusc = NuScenes(version=args.version, dataroot=DATAROOT, verbose=True)
    scene = nusc.scene[args.scene_idx]
    print(f'Scene: {scene["name"]}  ({scene["nbr_samples"]} samples)')

    out_path = args.out or os.path.join(
        REPO, 'video_out', f'nusc_{scene["name"]}_lane_{args.cam}.mp4')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── Collect all samples in scene ─────────────────────────────────────────
    samples = []
    token = scene['first_sample_token']
    while token:
        s = nusc.get('sample', token)
        samples.append(s)
        token = s['next']
    print(f'  {len(samples)} samples')

    # ── Phase 1: filter lane pts per frame, accumulate in world ──────────────
    print('[1/3] Filtering lane points ...')
    all_world_pts = []

    for s in samples:
        sd_lidar = nusc.get('sample_data', s['data']['LIDAR_TOP'])
        cs_lidar = nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])
        ep_lidar = nusc.get('ego_pose',           sd_lidar['ego_pose_token'])

        pc = LidarPointCloud.from_file(
            os.path.join(nusc.dataroot, sd_lidar['filename']))
        pts_lidar = pc.points[:3, :].T    # Nx3
        intensity  = pc.points[3, :]      # N  (0–255)

        lane_pts = filter_lane_pts(pts_lidar, intensity)
        if not len(lane_pts):
            continue

        # LiDAR → ego → world
        T_lidar_ego = make_T(cs_lidar['rotation'], cs_lidar['translation'])
        T_ego_world  = make_T(ep_lidar['rotation'], ep_lidar['translation'])
        T_lidar_world = T_ego_world @ T_lidar_ego

        all_world_pts.append(apply_T(T_lidar_world, lane_pts))

    if not all_world_pts:
        print('ERROR: No lane points found.')
        return

    raw_world = np.vstack(all_world_pts)
    print(f'  Accumulated {len(raw_world):,} world-frame lane pts')

    # Convert to local (first-frame ego) forward/lateral coords for clustering
    # Use first sample ego pose as reference "forward" axis
    ep0 = nusc.get('ego_pose',
                   nusc.get('sample_data',
                            samples[0]['data']['LIDAR_TOP'])['ego_pose_token'])
    T_world_ego0 = make_T(ep0['rotation'], ep0['translation'])
    T_ego0_world = np.linalg.inv(T_world_ego0)
    raw_ego0 = apply_T(T_ego0_world, raw_world)   # local frame

    # ── Phase 2: cluster + fit ────────────────────────────────────────────────
    print('[2/3] Clustering and fitting lanes ...')
    clusters = cluster_by_lateral(raw_ego0, min_pts=args.min_pts)
    print(f'  Found {len(clusters)} lane cluster(s)')

    fitted_local = []
    for c in clusters:
        r = fit_lane(c, degree=args.degree)
        if r is not None:
            fitted_local.append(r)

    if not fitted_local:
        print('ERROR: No lanes fitted.')
        return

    # Back to world frame for per-frame projection
    fitted_world = [apply_T(T_world_ego0, pts) for pts in fitted_local]
    print(f'  Fitted {len(fitted_world)} lane(s)')

    # ── Phase 3: project onto camera and write video ──────────────────────────
    print('[3/3] Rendering video ...')

    # Get image size from first frame
    sd0 = nusc.get('sample_data', samples[0]['data'][args.cam])
    img0 = cv2.imread(os.path.join(nusc.dataroot, sd0['filename']))
    H, W = img0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (W, H))

    for idx, s in enumerate(samples):
        sd_cam  = nusc.get('sample_data', s['data'][args.cam])
        cs_cam  = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
        ep_cam  = nusc.get('ego_pose',          sd_cam['ego_pose_token'])

        img = cv2.imread(os.path.join(nusc.dataroot, sd_cam['filename']))
        if img is None:
            continue

        # Build transforms: world → ego → cam
        T_ego_world = np.linalg.inv(make_T(ep_cam['rotation'], ep_cam['translation']))
        T_cam_ego   = np.linalg.inv(make_T(cs_cam['rotation'], cs_cam['translation']))
        T_world_ego_inv = T_ego_world   # world → ego
        T_cam_ego_inv   = T_cam_ego     # ego → cam

        K = np.array(cs_cam['camera_intrinsic'], dtype=np.float64)

        for li, pts_w in enumerate(fitted_world):
            color = LANE_COLORS[li % len(LANE_COLORS)]
            project_and_draw_lane(img, pts_w, T_world_ego_inv, T_cam_ego_inv,
                                  K, W, H, color)

        fid = f'{idx:03d}'
        cv2.putText(img, f'Frame {fid}  {scene["name"]}', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, f'Frame {fid}  {scene["name"]}', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
