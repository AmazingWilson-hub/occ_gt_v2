#!/usr/bin/env python3
"""
NuScenes 4-panel debug video:
  Top-left:     Raw LiDAR projected onto camera
  Top-right:    Filtered lane points (RANSAC + intensity)
  Bottom-left:  All-frame accumulated lane points (world → current ego)
  Bottom-right: Fitted polynomial lane lines

Usage:
    python3 lane_line/nusc_lane_debug_video.py \
        --scene_idx 0 --version v1.0-mini \
        --out video_out/nusc_lane_debug.mp4
"""

import os, argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

DATAROOT = '/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ'
REPO     = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

LANE_COLORS = [
    (  0, 255,   0), (  0, 255, 255), (255,   0,   0),
    (  0, 128, 255), (255,   0, 255), (255, 128,   0),
]

# ── Transforms ────────────────────────────────────────────────────────────────

def quat_to_rot(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def make_T(rot_wxyz, trans):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(rot_wxyz)
    T[:3,  3] = trans
    return T

def apply_T(T, pts):
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]

def lidar_to_cam_T(cs_lidar, ep_lidar, cs_cam, ep_cam):
    """Full chain: LiDAR → ego(lidar) → world → ego(cam) → cam."""
    T = (np.linalg.inv(make_T(cs_cam['rotation'],  cs_cam['translation']))  @
         np.linalg.inv(make_T(ep_cam['rotation'],  ep_cam['translation']))  @
         make_T(ep_lidar['rotation'], ep_lidar['translation'])               @
         make_T(cs_lidar['rotation'], cs_lidar['translation']))
    return T

def project_pinhole(pts_cam, K, W, H):
    """Nx3 camera-frame pts → (u, v, mask_in_image)."""
    front = pts_cam[:, 2] > 0.1
    u = np.full(len(pts_cam), -1.0)
    v = np.full(len(pts_cam), -1.0)
    p = pts_cam[front]
    u[front] = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
    v[front] = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    in_img = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, in_img

# ── Lane filter ───────────────────────────────────────────────────────────────

def filter_lane_pts(pts, intensity):
    from sklearn import linear_model
    from sklearn.cluster import KMeans

    mask = ((pts[:, 0] > 2)   & (pts[:, 0] < 60) &
            (pts[:, 1] > -10) & (pts[:, 1] < 10))
    pts  = pts[mask];  intensity = intensity[mask]
    if len(pts) < 20:
        return np.zeros((0, 3))

    try:
        ransac = linear_model.RANSACRegressor(
            linear_model.LinearRegression(),
            min_samples=5, residual_threshold=0.2, max_trials=150)
        ransac.fit(pts[:, :2], pts[:, 2])
        ground = ransac.inlier_mask_
    except Exception:
        ground = pts[:, 2] < -1.0

    pts_g  = pts[ground]
    inty_g = intensity[ground] / 255.0
    if len(pts_g) < 10:
        return np.zeros((0, 3))

    i_range = inty_g.max() - inty_g.min()
    if i_range < 0.1:
        return np.zeros((0, 3))

    n_clusters = 2 if i_range > 0.4 else 3
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(inty_g.reshape(-1, 1))
    keep = labels != np.argmax(np.bincount(labels))
    return pts_g[keep]

# ── Lane fitting ──────────────────────────────────────────────────────────────

def cluster_by_lateral(pts, min_pts=30, bw=0.25):
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    ys = pts[:, 1]
    y_grid = np.linspace(ys.min() - 1, ys.max() + 1, 2000)
    dy = y_grid[1] - y_grid[0]
    kde = gaussian_kde(ys, bw_method=bw / ys.std())
    density = kde(y_grid)
    peaks, _ = find_peaks(density, distance=1.0 / dy, prominence=density.max() * 0.01)
    if not len(peaks):
        return [pts] if len(pts) >= min_pts else []
    cut_ys = [y_grid[lo + np.argmin(density[lo:hi + 1])]
              for lo, hi in zip(peaks[:-1], peaks[1:])]
    order = np.argsort(ys)
    groups = np.split(order, np.searchsorted(ys[order], cut_ys))
    return [pts[g] for g in groups if len(g) >= min_pts]

def fit_lane(pts, degree=3, n_samples=300):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    if x.max() - x.min() < 5:
        return None
    x_s = np.linspace(x.min(), x.max(), n_samples)
    y_s = np.polyval(np.polyfit(x, y, degree), x_s)
    z_s = np.polyval(np.polyfit(x, z, degree), x_s)
    if n_samples > 21:
        y_s = savgol_filter(y_s, 21, 2)
        z_s = savgol_filter(z_s, 21, 2)
    return np.stack([x_s, y_s, z_s], axis=1)

# ── Draw helpers ──────────────────────────────────────────────────────────────

def draw_pts_depth(img, u, v, mask, depth, max_d, radius=3):
    z_norm = np.clip(depth[mask] / max_d, 0, 1)
    for i, (ui, vi) in enumerate(zip(u[mask], v[mask])):
        c = int(z_norm[i] * 255)
        cv2.circle(img, (int(ui), int(vi)), radius, (255 - c, c, 0), -1)

def draw_pts_white(img, u, v, mask, radius=4):
    for ui, vi in zip(u[mask], v[mask]):
        cv2.circle(img, (int(ui), int(vi)), radius, (255, 255, 255), -1)

def draw_lanes(img, fitted_world, T_world_to_cam, K, W, H):
    for li, pts_w in enumerate(fitted_world):
        pts_cam = apply_T(T_world_to_cam, pts_w)
        # range filter
        pts_ego_approx = pts_cam  # already in cam frame for range check is tricky;
        # just use z > 0 and reasonable angle
        z = pts_cam[:, 2]
        tx = np.abs(pts_cam[:, 0]) / np.maximum(z, 1e-6)
        ty = np.abs(pts_cam[:, 1]) / np.maximum(z, 1e-6)
        ok = (z > 0.1) & (tx < 2.0) & (ty < 2.0)
        u, v, in_b = project_pinhole(pts_cam, K, W, H)
        in_b &= ok
        color = LANE_COLORS[li % len(LANE_COLORS)]
        for i in range(len(pts_cam) - 1):
            if in_b[i] and in_b[i + 1]:
                cv2.line(img, (int(u[i]), int(v[i])), (int(u[i+1]), int(v[i+1])),
                         color, 3, cv2.LINE_AA)

def label(img, text, pos=(30, 55)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (0, 0, 0), 2, cv2.LINE_AA)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_idx', type=int, default=0)
    parser.add_argument('--version',   default='v1.0-mini')
    parser.add_argument('--cam',       default='CAM_FRONT')
    parser.add_argument('--out',       default=None)
    parser.add_argument('--fps',       type=int, default=2)
    parser.add_argument('--max_dist',  type=float, default=60.0)
    parser.add_argument('--degree',    type=int,   default=3)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud

    nusc  = NuScenes(version=args.version, dataroot=DATAROOT, verbose=True)
    scene = nusc.scene[args.scene_idx]
    print(f'Scene: {scene["name"]}')

    out_path = args.out or os.path.join(
        REPO, 'video_out', f'nusc_{scene["name"]}_debug.mp4')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Collect samples
    samples, tok = [], scene['first_sample_token']
    while tok:
        s = nusc.get('sample', tok); samples.append(s); tok = s['next']
    print(f'{len(samples)} samples')

    # ── Pass 1: filter + accumulate lane points in world frame ───────────────
    print('[1/3] Filtering and accumulating ...')
    frame_data  = []   # per-frame cache
    world_pts   = []   # accumulated lane pts in world

    for s in samples:
        sd_l  = nusc.get('sample_data', s['data']['LIDAR_TOP'])
        cs_l  = nusc.get('calibrated_sensor', sd_l['calibrated_sensor_token'])
        ep_l  = nusc.get('ego_pose',           sd_l['ego_pose_token'])
        sd_c  = nusc.get('sample_data', s['data'][args.cam])
        cs_c  = nusc.get('calibrated_sensor', sd_c['calibrated_sensor_token'])
        ep_c  = nusc.get('ego_pose',           sd_c['ego_pose_token'])

        pc       = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_l['filename']))
        pts_l    = pc.points[:3, :].T
        inty     = pc.points[3, :]

        lane_ego = filter_lane_pts(pts_l, inty)   # in LiDAR frame (≈ego)

        # LiDAR → world for accumulation
        T_l2w = make_T(ep_l['rotation'], ep_l['translation']) @ make_T(cs_l['rotation'], cs_l['translation'])
        if len(lane_ego):
            world_pts.append(apply_T(T_l2w, lane_ego))

        frame_data.append({
            'pts_l': pts_l, 'inty': inty,
            'lane_ego': lane_ego,
            'cs_l': cs_l, 'ep_l': ep_l,
            'cs_c': cs_c, 'ep_c': ep_c,
            'img_path': os.path.join(nusc.dataroot, sd_c['filename']),
        })
        print(f'  {sd_l["filename"][-12:]}  lane pts: {len(lane_ego)}')

    all_world = np.vstack(world_pts) if world_pts else np.zeros((0, 3))
    print(f'  Total world lane pts: {len(all_world):,}')

    # ── Pass 2: fit lanes ────────────────────────────────────────────────────
    print('[2/3] Fitting lanes ...')
    fitted_world = []
    if len(all_world) >= 50:
        # Cluster in first-frame ego space
        ep0   = frame_data[0]['ep_l']
        cs0   = frame_data[0]['cs_l']
        T_w2e0 = np.linalg.inv(make_T(ep0['rotation'], ep0['translation']) @
                                make_T(cs0['rotation'], cs0['translation']))
        local  = apply_T(T_w2e0, all_world)
        clusters = cluster_by_lateral(local, min_pts=30)
        print(f'  {len(clusters)} cluster(s)')
        T_e02w = np.linalg.inv(T_w2e0)
        for c in clusters:
            r = fit_lane(c, degree=args.degree)
            if r is not None:
                fitted_world.append(apply_T(T_e02w, r))
        print(f'  {len(fitted_world)} fitted lane(s)')

    # ── Pass 3: render 2×2 video ─────────────────────────────────────────────
    print('[3/3] Rendering ...')
    img0 = cv2.imread(frame_data[0]['img_path'])
    H, W = img0.shape[:2]
    pH, pW = H // 2, W // 2   # panel size (scale each panel to half)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (pW * 2, pH * 2))

    for idx, (s, fd) in enumerate(zip(samples, frame_data)):
        base = cv2.imread(fd['img_path'])
        if base is None:
            continue

        T_l2cam = lidar_to_cam_T(fd['cs_l'], fd['ep_l'], fd['cs_c'], fd['ep_c'])
        K       = np.array(fd['cs_c']['camera_intrinsic'], dtype=np.float64)

        pts_cam_all  = apply_T(T_l2cam, fd['pts_l'])
        depth_all    = np.linalg.norm(fd['pts_l'], axis=1)

        # ── Panel A: raw LiDAR ──────────────────────────────────────────────
        pa = base.copy()
        u, v, in_b = project_pinhole(pts_cam_all, K, W, H)
        near = in_b & (depth_all < args.max_dist)
        draw_pts_depth(pa, u, v, near, depth_all, args.max_dist, radius=2)
        label(pa, f'[{idx:03d}] Raw LiDAR')

        # ── Panel B: filtered lane pts ──────────────────────────────────────
        pb = base.copy()
        if len(fd['lane_ego']):
            pts_lane_cam = apply_T(T_l2cam, fd['lane_ego'])
            ul, vl, in_bl = project_pinhole(pts_lane_cam, K, W, H)
            draw_pts_white(pb, ul, vl, in_bl, radius=5)
        label(pb, f'[{idx:03d}] Filtered lanes')

        # ── Panel C: accumulated lane pts ───────────────────────────────────
        pc_img = base.copy()
        if len(all_world):
            T_world2cam = (np.linalg.inv(make_T(fd['cs_c']['rotation'], fd['cs_c']['translation'])) @
                           np.linalg.inv(make_T(fd['ep_c']['rotation'], fd['ep_c']['translation'])))
            acc_cam = apply_T(T_world2cam, all_world)
            ua, va, in_ba = project_pinhole(acc_cam, K, W, H)
            draw_pts_white(pc_img, ua, va, in_ba, radius=3)
        label(pc_img, f'[{idx:03d}] Accumulated')

        # ── Panel D: fitted lane lines ──────────────────────────────────────
        pd_img = base.copy()
        if fitted_world:
            T_world2cam = (np.linalg.inv(make_T(fd['cs_c']['rotation'], fd['cs_c']['translation'])) @
                           np.linalg.inv(make_T(fd['ep_c']['rotation'], fd['ep_c']['translation'])))
            draw_lanes(pd_img, fitted_world, T_world2cam, K, W, H)
        label(pd_img, f'[{idx:03d}] Fitted lanes')

        # ── Combine 2×2 ─────────────────────────────────────────────────────
        row0 = np.hstack([cv2.resize(pa, (pW, pH)), cv2.resize(pb, (pW, pH))])
        row1 = np.hstack([cv2.resize(pc_img, (pW, pH)), cv2.resize(pd_img, (pW, pH))])
        frame = np.vstack([row0, row1])
        writer.write(frame)

    writer.release()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
