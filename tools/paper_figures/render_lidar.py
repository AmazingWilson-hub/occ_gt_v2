#!/usr/bin/env python3
"""
Render raw nuScenes LiDAR point cloud at the same chase-cam angle as render_occ.py.
Useful for comparison: sparse raw LiDAR vs dense occupancy grid.

Usage:
    python3 render_lidar.py \
        --token 88449a5cb1644a199c1c11f6ac034867 \
        --out_dir paper_out
"""

import argparse
import os
import sys
import numpy as np
import cv2

# ── nuScenes constants ────────────────────────────────────────────────────────
DATAROOT   = '/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ'
NS_VERSION = 'v1.0-mini'

# Same view parameters as render_occ.py render_3d()
VIEW_RANGE = 35.0   # ±35m around ego
ELEV_DEG   = 28
OCC_W      = 1200
OCC_H      = 700

# nuScenes lidarseg → Occ3D 17-class mapping (same as generate.py)
# nuScenes has 32 classes; we remap to Occ3D 17+free
NUSCENES_TO_OCC3D = {
    0:  17,  # noise         → free
    1:  17,  # animal        → free
    2:   7,  # human pedestrian adult
    3:   7,  # human pedestrian child
    4:   7,  # human pedestrian construction worker
    5:   7,  # human pedestrian personal mobility
    6:   7,  # human pedestrian police officer
    7:   7,  # human pedestrian stroller
    8:   7,  # human pedestrian wheelchair
    9:   1,  # movable object barrier
    10:  2,  # vehicle bicycle
    11:  9,  # vehicle bus bendy
    12:  3,  # vehicle bus rigid
    13:  4,  # vehicle car
    14:  5,  # vehicle construction
    15: 17,  # vehicle emergency ambulance → free
    16: 17,  # vehicle emergency police → free
    17:  6,  # vehicle motorcycle
    18: 15,  # static manmade
    19: 16,  # vegetation
    20: 11,  # flat driveable surface
    21: 12,  # flat other
    22: 13,  # flat sidewalk
    23: 14,  # flat terrain
    24: 15,  # static manmade (duplicate)
    25: 15,  # static other object → manmade
    26: 10,  # vehicle truck
    27:  9,  # vehicle trailer
    28:  8,  # vehicle traffic cone
    29:  5,  # vehicle construction → construction_vehicle
    30: 17,  # free
    31: 17,  # free
}

# Occ3D RGB color table (same as render_occ.py)
OCC3D_COLORS = np.array([
    [ 80,  80,  80],   # 0:  others
    [112, 128, 144],   # 1:  barrier
    [220,  20,  60],   # 2:  bicycle
    [255, 158,   0],   # 3:  bus
    [255, 158,   0],   # 4:  car
    [233, 150,  70],   # 5:  construction_vehicle
    [255, 100,  50],   # 6:  motorcycle
    [  0,   0, 230],   # 7:  pedestrian
    [255, 215,   0],   # 8:  traffic_cone
    [255, 140,   0],   # 9:  trailer
    [205,  92,  92],   # 10: truck
    [  0, 207, 191],   # 11: driveable_surface
    [135, 206, 235],   # 12: other_flat
    [ 75,   0,  75],   # 13: sidewalk
    [ 50, 180, 100],   # 14: terrain
    [222, 184, 135],   # 15: manmade
    [  0, 175,   0],   # 16: vegetation
    [200, 200, 200],   # 17: free → light grey for LiDAR
], dtype=np.uint8)


def quat_to_rot(q):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def load_lidar_in_ego(nusc, sample_token):
    """
    Load LiDAR point cloud for a sample and transform to ego frame.
    Returns (N,3) xyz array and (N,) semantic label array (Occ3D class indices).
    """
    from nuscenes.utils.data_classes import LidarPointCloud

    sample   = nusc.get('sample', sample_token)
    sd_token = sample['data']['LIDAR_TOP']
    sd_rec   = nusc.get('sample_data', sd_token)

    # Load raw points (x,y,z,intensity,ring)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_rec['filename']))
    pts = pc.points[:3, :]   # (3, N)

    # sensor → ego
    cs  = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    R_s = quat_to_rot(cs['rotation'])
    t_s = np.array(cs['translation'])
    pts = R_s @ pts + t_s[:, None]   # (3, N)

    pts = pts.T   # (N, 3)

    # Semantic labels from lidarseg (if available)
    try:
        seg_file = os.path.join(nusc.dataroot,
                                nusc.get('lidarseg', sd_token)['filename'])
        raw_labels = np.fromfile(seg_file, dtype=np.uint8)
        occ_labels = np.array([NUSCENES_TO_OCC3D.get(int(l), 17)
                                for l in raw_labels], dtype=np.uint8)
    except Exception:
        # fallback: color by height
        z_norm  = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-6)
        occ_labels = np.full(len(pts), 17, dtype=np.uint8)   # placeholder; height coloring applied later

    return pts, occ_labels


def render_lidar_chase_cam(pts, labels, out_path,
                           elev_deg=ELEV_DEG, occ_w=OCC_W, occ_h=OCC_H,
                           view_range=VIEW_RANGE, z_offset=0.0, pt_size=2):
    """
    Render LiDAR points with the same chase-cam projection as render_occ.py render_3d().
    Points are colored by Occ3D semantic class (or height if labels are all 17).
    """
    # filter to view range
    in_range = ((pts[:, 0] >= -view_range) & (pts[:, 0] <= view_range) &
                (pts[:, 1] >= -view_range) & (pts[:, 1] <= view_range))
    pts    = pts[in_range]
    labels = labels[in_range]

    px = pts[:, 0]
    py = pts[:, 1]
    pz = pts[:, 2] + z_offset

    # Color: use Occ3D color table; for non-semantic (all free), use z-height
    if np.all(labels == 17):
        z_norm  = (pz - pz.min()) / (pz.max() - pz.min() + 1e-6)
        base_c  = np.array([[50, 150, 220]], dtype=np.float32)
        colors  = np.clip(base_c * (0.4 + 0.6 * z_norm)[:, None], 0, 255).astype(np.uint8)
    else:
        colors = OCC3D_COLORS[labels].astype(np.float32)
        z_norm = (pz - pz.min()) / (pz.max() - pz.min() + 1e-6)
        colors = np.clip(colors * (0.55 + 0.45 * z_norm)[:, None], 0, 255).astype(np.uint8)

    # Chase-cam projection (same as render_occ.py)
    elev  = np.radians(elev_deg)
    se, ce = np.sin(elev), np.cos(elev)
    scale  = occ_w / (2.0 * view_range)

    x_2d  = -py
    y_2d  =  px * se + pz * ce
    depth = -px * ce + pz * se

    ego_sx = occ_w // 2
    ego_sy = int(occ_h * 0.70)
    x_scr  = (x_2d * scale + ego_sx).astype(np.int32)
    y_scr  = (ego_sy - y_2d * scale).astype(np.int32)

    # back-to-front sort
    order   = np.argsort(depth)
    x_s, y_s, c_s = x_scr[order], y_scr[order], colors[order]

    canvas = np.full((occ_h, occ_w, 3), 255, dtype=np.uint8)

    # Draw each point as a small square
    B = pt_size
    oy, ox = np.mgrid[0:B, 0:B]
    offsets = np.stack([ox.ravel(), oy.ravel()], axis=1)
    n_v, n_o = len(x_s), len(offsets)

    x_all = np.repeat(x_s, n_o) + np.tile(offsets[:, 0], n_v)
    y_all = np.repeat(y_s, n_o) + np.tile(offsets[:, 1], n_v)
    c_all = np.repeat(c_s, n_o, axis=0)

    valid  = (x_all >= 0) & (x_all < occ_w) & (y_all >= 0) & (y_all < occ_h)
    canvas[y_all[valid], x_all[valid]] = c_all[valid]

    cv2.imwrite(out_path, canvas[:, :, ::-1])
    print(f'LiDAR saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token',
                        default='88449a5cb1644a199c1c11f6ac034867',
                        help='nuScenes sample token')
    parser.add_argument('--out_dir', default='paper_out')
    parser.add_argument('--pt_size', type=int, default=2,
                        help='Point size in pixels (default 2)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading nuScenes...')
    sys.path.insert(0, os.path.dirname(__file__))
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=NS_VERSION, dataroot=DATAROOT, verbose=False)

    print(f'Loading LiDAR for token: {args.token}')
    pts, labels = load_lidar_in_ego(nusc, args.token)
    print(f'Points: {len(pts)}  (semantic labels available: {not np.all(labels==17)})')

    out_path = os.path.join(args.out_dir, 'lidar_chase_cam.png')
    render_lidar_chase_cam(pts, labels, out_path, pt_size=args.pt_size)


if __name__ == '__main__':
    main()
