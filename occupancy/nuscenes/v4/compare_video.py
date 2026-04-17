#!/usr/bin/env python3
"""
compare_video.py — Three-way sweep comparison video for NuScenes

Layout:
  Row 1: CAM_FRONT_LEFT | CAM_FRONT | CAM_FRONT_RIGHT
  Row 2: CAM_BACK_LEFT  | CAM_BACK  | CAM_BACK_RIGHT
  Row 3: CVPR GT        | KISS-SLAM all-10 | KISS-SLAM road-40 rest-10

Usage:
  # First generate the two KISS-SLAM outputs, then:
  python3 compare_video.py --scene scene-0061 --backend kiss_slam

  # Custom paths:
  python3 compare_video.py \
      --dir_a /path/to/gts/scene-0061 \
      --dir_b /path/to/kiss_slam_all10/scene-0061 \
      --dir_c /path/to/v4/kiss_slam/scene-0061
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'trajectory', 'egotest', 'cvpr_format_occ_gen_egotest'))

DATAROOT = '/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ'

# --- Occ3D colors (BGR) ---
OCC3D_COLORS_BGR = np.array([
    [  0,   0,   0],   # 0:  noise
    [112, 128, 144],   # 1:  barrier
    [ 60,  20, 220],   # 2:  bicycle
    [  0, 158, 255],   # 3:  bus
    [  0, 158, 255],   # 4:  car
    [  0, 158, 255],   # 5:  construction_vehicle
    [  0, 158, 255],   # 6:  motorcycle
    [230,   0,   0],   # 7:  pedestrian
    [112, 128, 144],   # 8:  traffic_cone
    [  0, 158, 255],   # 9:  trailer
    [  0, 158, 255],   # 10: truck
    [191, 207,   0],   # 11: driveable_surface
    [247, 207,  70],   # 12: other_flat
    [ 75,   0,  75],   # 13: sidewalk
    [191, 207,   0],   # 14: terrain
    [135, 184, 222],   # 15: manmade
    [  0, 175,   0],   # 16: vegetation
    [255, 255, 255],   # 17: free_space (not rendered)
], dtype=np.uint8)

VOXEL_SIZE = 0.4
BG_COLOR   = (25, 25, 25)

CAM_NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']
CAM_LABELS = ['Front Left', 'Front', 'Front Right',
              'Back Left',  'Back',  'Back Right']


# ---------------------------------------------------------------------------
# NuScenes helpers
# ---------------------------------------------------------------------------

def load_nuscenes():
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version='v1.0-mini', dataroot=DATAROOT)


def get_scene_tokens(nusc, scene_name):
    """Return ordered list of (sample_token, {cam_name: img_path}) for scene."""
    scene = next(s for s in nusc.scene if s['name'] == scene_name)
    entries = []
    token = scene['first_sample_token']
    while token:
        sample = nusc.get('sample', token)
        cam_paths = {}
        for cam in CAM_NAMES:
            sd = nusc.get('sample_data', sample['data'][cam])
            cam_paths[cam] = os.path.join(nusc.dataroot, sd['filename'])
        entries.append((token, cam_paths))
        token = sample['next'] if sample['next'] else None
    return entries


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _build_block_offsets(block):
    offsets, brightness = [], []
    for dy in range(block):
        face_b = 1.25 - 0.55 * (dy / max(block - 1, 1))
        for dx in range(block):
            is_edge = (dx == 0 or dx == block - 1 or dy == 0 or dy == block - 1)
            offsets.append((dx, dy))
            brightness.append(face_b * 0.45 if is_edge else face_b)
    return (np.array(offsets, dtype=np.int32),
            np.array(brightness, dtype=np.float32))


def render_chase_cam(grid, elev_deg, occ_h, occ_w):
    """Chase-cam 3D render (same as U5 pipeline)."""
    occupied = (grid != 17) & (grid != 0)
    if not np.any(occupied):
        return np.full((occ_h, occ_w, 3), BG_COLOR, dtype=np.uint8)

    xs, ys, zs = np.where(occupied)
    px = (xs - 100.0) * VOXEL_SIZE
    py = (ys - 100.0) * VOXEL_SIZE
    pz = zs * VOXEL_SIZE + (-1.0)

    labels = grid[xs, ys, zs]
    colors = OCC3D_COLORS_BGR[labels].astype(np.float64)
    z_norm = (pz - pz.min()) / (pz.max() - pz.min() + 1e-6)
    colors = np.clip(colors * (0.6 + 0.4 * z_norm)[:, np.newaxis], 0, 255).astype(np.uint8)

    # Ego car (green voxels)
    car_ex = np.arange(94, 106)
    car_ey = np.arange(97, 103)
    car_ez = np.arange(5, 9)
    ego_grid = np.stack(np.meshgrid(car_ex, car_ey, car_ez, indexing='ij'), axis=-1).reshape(-1, 3)
    ego_px = (ego_grid[:, 0] - 100.0) * VOXEL_SIZE
    ego_py = (ego_grid[:, 1] - 100.0) * VOXEL_SIZE
    ego_pz = ego_grid[:, 2] * VOXEL_SIZE + (-1.0)
    ego_z_norm = (ego_pz - ego_pz.min()) / (ego_pz.max() - ego_pz.min() + 1e-6)
    ego_colors = np.clip(
        np.array([50, 255, 0], dtype=np.float64) * (0.6 + 0.4 * ego_z_norm)[:, np.newaxis],
        0, 255).astype(np.uint8)

    px = np.concatenate([px, ego_px])
    py = np.concatenate([py, ego_py])
    pz = np.concatenate([pz, ego_pz])
    colors = np.concatenate([colors, ego_colors])

    elev = np.radians(elev_deg)
    se, ce = np.sin(elev), np.cos(elev)
    x_2d  =  -py
    y_2d  =   px * se + pz * ce
    depth = -px * ce + pz * se

    ego_frac_x, ego_frac_y, margin = 0.50, 0.60, 0.04
    ego_x2d, ego_y2d = 0.0, 0.0
    scale = min(
        occ_w * (ego_frac_x - margin)       / max(ego_x2d - x_2d.min(),   0.1),
        occ_w * (1 - ego_frac_x - margin)   / max(x_2d.max() - ego_x2d,   0.1),
        occ_h * (ego_frac_y - margin)        / max(y_2d.max() - ego_y2d,   0.1),
        occ_h * (1 - ego_frac_y - margin)    / max(ego_y2d - y_2d.min(),   0.1),
    )
    ego_scr_x = int(occ_w * ego_frac_x)
    ego_scr_y = int(occ_h * ego_frac_y)
    x_screen = ((x_2d - ego_x2d) * scale + ego_scr_x).astype(np.int32)
    y_screen = (ego_scr_y - (y_2d - ego_y2d) * scale).astype(np.int32)

    order = np.argsort(depth)
    x_s, y_s, c_s = x_screen[order], y_screen[order], colors[order]

    block = max(4, int(np.ceil(VOXEL_SIZE * scale * 1.15)))
    offsets, bf = _build_block_offsets(block)
    n_v, n_o = len(x_s), len(offsets)

    x_all = np.repeat(x_s, n_o) + np.tile(offsets[:, 0], n_v)
    y_all = np.repeat(y_s, n_o) + np.tile(offsets[:, 1], n_v)
    c_all = np.clip(
        np.repeat(c_s, n_o, axis=0).astype(np.float64) * np.tile(bf, n_v)[:, np.newaxis],
        0, 255).astype(np.uint8)

    valid = (x_all >= 0) & (x_all < occ_w) & (y_all >= 0) & (y_all < occ_h)
    canvas = np.full((occ_h, occ_w, 3), BG_COLOR, dtype=np.uint8)
    canvas[y_all[valid], x_all[valid]] = c_all[valid]
    return canvas


def render_occ_panel(npz_path, label, elev_deg, occ_h, occ_w):
    """Load npz and render one 3D panel with title."""
    if npz_path and os.path.exists(npz_path):
        grid = np.load(npz_path)['semantics']
    else:
        grid = np.ones((200, 200, 16), dtype=np.uint8) * 17  # empty

    img = render_chase_cam(grid, elev_deg, occ_h, occ_w)

    # Title bar at top
    bar_h = 36
    bar = np.zeros((bar_h, occ_w, 3), dtype=np.uint8)
    cv2.putText(bar, label, (10, bar_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    return np.vstack([bar, img])  # (occ_h + bar_h) x occ_w


def render_cam_strip(cam_paths, strip_w, row_h):
    """Render 6 cameras in 2 rows of 3."""
    cam_w = strip_w // 3
    strip = np.full((row_h * 2, strip_w, 3), 30, dtype=np.uint8)

    for i, (cam, label) in enumerate(zip(CAM_NAMES, CAM_LABELS)):
        row, col = divmod(i, 3)
        y0 = row * row_h
        x0 = col * cam_w
        path = cam_paths.get(cam)
        if path and os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (cam_w, row_h))
                cv2.putText(img, label, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                strip[y0:y0 + row_h, x0:x0 + cam_w] = img

    cv2.line(strip, (0, row_h - 1), (strip_w, row_h - 1), (60, 60, 60), 1)
    for col in range(1, 3):
        cv2.line(strip, (col * cam_w, 0), (col * cam_w, row_h * 2), (60, 60, 60), 1)
    return strip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Three-way sweep comparison video')
    parser.add_argument('--scene',   default='scene-0061')
    parser.add_argument('--backend', default='kiss_slam')

    gts_default = os.path.join(os.path.dirname(__file__), '..', 'data', 'nuscenes_occ', 'gts')
    v4_default  = os.path.join(os.path.dirname(__file__), 'output')

    # Three occupancy directories (scene level)
    parser.add_argument('--dir_a', default=None,
                        help='CVPR GT dir (scene level). Default: data/nuscenes_occ/gts/<scene>')
    parser.add_argument('--dir_b', default=None,
                        help='KISS-SLAM all-10 output dir (scene level)')
    parser.add_argument('--dir_c', default=None,
                        help='KISS-SLAM road-40 rest-10 output dir (scene level, V4 default)')

    parser.add_argument('--fps',     type=int,   default=4)
    parser.add_argument('--elev',    type=float, default=28)
    parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(__file__), 'video_out'))
    parser.add_argument('--total_w', type=int, default=1920)
    parser.add_argument('--occ_h',   type=int, default=480)
    args = parser.parse_args()

    # Resolve default paths
    dir_a = args.dir_a or os.path.join(gts_default, args.scene)
    dir_b = args.dir_b or os.path.join(v4_default, f'{args.backend}_all10', args.backend, args.scene)
    dir_c = args.dir_c or os.path.join(v4_default, args.backend, args.scene)

    labels = ['CVPR GT', 'KISS-SLAM  all=10', 'KISS-SLAM  road=40, others=10']
    dirs   = [dir_a, dir_b, dir_c]

    for d, l in zip(dirs, labels):
        exists = d and os.path.isdir(d)
        print(f"  [{('OK' if exists else 'MISSING')}] {l}: {d}")

    nusc = load_nuscenes()
    scene_entries = get_scene_tokens(nusc, args.scene)
    print(f"\nScene {args.scene}: {len(scene_entries)} frames")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.scene}_{args.backend}_compare.mp4')

    # Layout dimensions
    total_w  = args.total_w
    cam_w    = total_w // 3
    cam_row_h = int(cam_w * 900 / 1600)   # approx NuScenes cam aspect
    cam_strip_h = cam_row_h * 2
    occ_panel_w = total_w // 3
    bar_h    = 36
    occ_panel_h = args.occ_h + bar_h
    total_h  = cam_strip_h + occ_panel_h

    print(f"Canvas: {total_w}x{total_h}  (cam {cam_w}x{cam_row_h}x2 + occ {occ_panel_w}x{args.occ_h})")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (total_w, total_h))

    for sample_token, cam_paths in tqdm(scene_entries, desc='Rendering'):
        # --- Camera strip ---
        cam_strip = render_cam_strip(cam_paths, total_w, cam_row_h)

        # --- Three occupancy panels ---
        panels = []
        for d, lbl in zip(dirs, labels):
            npz = os.path.join(d, sample_token, 'labels.npz') if d else None
            panels.append(render_occ_panel(npz, lbl, args.elev, args.occ_h, occ_panel_w))

        # Align panel heights (all same, just stack horizontally)
        occ_row = np.hstack(panels)   # (occ_panel_h, total_w, 3)

        # Dividers between panels
        for col in range(1, 3):
            x = col * occ_panel_w
            cv2.line(occ_row, (x, 0), (x, occ_panel_h), (80, 80, 80), 2)

        # Frame info
        cv2.putText(occ_row, f'{args.scene}  token: {sample_token[:8]}',
                    (total_w - 420, occ_panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        frame = np.vstack([cam_strip, occ_row])
        writer.write(frame)

    writer.release()
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
