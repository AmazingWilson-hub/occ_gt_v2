#!/usr/bin/env python3
"""
compare_video.py — ELAN three-way occupancy comparison video

Layout:
  Top:    Front camera (full width)
  Bottom: raw (no vehicles) | heuristic | seg (semantic)
"""

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

OCC3D_COLORS_BGR = np.array([
    [  0,   0,   0],   # 0:  noise
    [144, 128, 112],   # 1:  barrier
    [ 43, 191, 235],   # 2:  bicycle
    [ 33,  91, 255],   # 3:  bus
    [ 60,  20, 220],   # 4:  car
    [ 33,  91, 255],   # 5:  construction_vehicle
    [  0,  61, 135],   # 6:  motorcycle
    [ 76,   0,  75],   # 7:  pedestrian
    [144, 128, 112],   # 8:  traffic_cone
    [ 33,  91, 255],   # 9:  trailer
    [ 33,  91, 255],   # 10: truck
    [248, 150, 215],   # 11: driveable_surface  (粉紫)
    [ 70, 206, 247],   # 12: other_flat
    [ 70, 206, 247],   # 13: sidewalk
    [152, 251, 152],   # 14: terrain
    [ 70, 206, 247],   # 15: manmade
    [152, 251, 152],   # 16: vegetation
    [255, 255, 255],   # 17: free_space (skip)
], dtype=np.uint8)

VOXEL_SIZE   = 0.4
BG_COLOR     = (50, 50, 50)
BLOCK_SIZE   = 6        # fixed voxel pixel size
VIEW_RANGE   = 35.0    # metres — overhead view shows ±35m around ego

# Pre-compute ego car voxels (constant every frame)
_car_ex = np.arange(94, 106)
_car_ey = np.arange(97, 103)
_car_ez = np.arange(5, 9)
_ego_grid = np.stack(np.meshgrid(_car_ex, _car_ey, _car_ez, indexing='ij'), axis=-1).reshape(-1, 3)
EGO_PX = (_ego_grid[:, 0] - 100.0) * VOXEL_SIZE
EGO_PY = (_ego_grid[:, 1] - 100.0) * VOXEL_SIZE
EGO_PZ = _ego_grid[:, 2] * VOXEL_SIZE + (-2.0)
_ego_z_norm = (EGO_PZ - EGO_PZ.min()) / (EGO_PZ.max() - EGO_PZ.min() + 1e-6)
EGO_COLORS = np.clip(
    np.array([50, 255, 0], dtype=np.float64) * (0.35 + 0.65 * _ego_z_norm)[:, np.newaxis],
    0, 255).astype(np.uint8)

# Pre-compute block offsets
_BLOCK_OFFSETS, _BLOCK_BF = None, None


def _build_block_offsets(block):
    """Top face bright, bottom-edge darker for 3D cube feel."""
    offsets, brightness = [], []
    for dy in range(block):
        # top rows bright, bottom rows darker
        face_b = 1.0 - 0.5 * (dy / max(block - 1, 1))
        for dx in range(block):
            is_edge = (dx == 0 or dx == block - 1 or dy == block - 1)
            offsets.append((dx, dy))
            brightness.append(face_b * 0.5 if is_edge else face_b)
    return (np.array(offsets, dtype=np.int32),
            np.array(brightness, dtype=np.float32))


def render_chase_cam(grid, elev_deg, occ_h, occ_w):
    global _BLOCK_OFFSETS, _BLOCK_BF
    if _BLOCK_OFFSETS is None:
        _BLOCK_OFFSETS, _BLOCK_BF = _build_block_offsets(BLOCK_SIZE)

    occupied = (grid != 17) & (grid != 0)
    if not np.any(occupied):
        return np.full((occ_h, occ_w, 3), BG_COLOR, dtype=np.uint8)

    xs, ys, zs = np.where(occupied)
    px = (xs - 100.0) * VOXEL_SIZE
    py = (ys - 100.0) * VOXEL_SIZE
    pz = zs * VOXEL_SIZE + (-2.0)

    labels = grid[xs, ys, zs]
    colors = OCC3D_COLORS_BGR[labels].astype(np.float32)
    z_norm = (pz - pz.min()) / (pz.max() - pz.min() + 1e-6)
    colors = np.clip(colors * (0.35 + 0.65 * z_norm)[:, np.newaxis], 0, 255).astype(np.uint8)

    # Use pre-computed ego car
    px = np.concatenate([px, EGO_PX])
    py = np.concatenate([py, EGO_PY])
    pz = np.concatenate([pz, EGO_PZ])
    colors = np.concatenate([colors, EGO_COLORS])

    elev = np.radians(elev_deg)
    se, ce = np.sin(elev), np.cos(elev)
    x_2d  = -py
    y_2d  =  px * se + pz * ce
    depth = -px * ce + pz * se

    scale = occ_w / (2.0 * VIEW_RANGE)

    ego_scr_x = occ_w // 2
    ego_scr_y = int(occ_h * 0.62)

    x_screen = (x_2d * scale + ego_scr_x).astype(np.int32)
    y_screen = (ego_scr_y - y_2d * scale).astype(np.int32)

    # Pre-filter to screen bounds before expensive expand (major speedup)
    margin = BLOCK_SIZE
    visible = ((x_screen >= -margin) & (x_screen < occ_w + margin) &
               (y_screen >= -margin) & (y_screen < occ_h + margin))
    x_screen = x_screen[visible]
    y_screen  = y_screen[visible]
    colors    = colors[visible]
    depth     = depth[visible]

    order = np.argsort(depth)
    x_s, y_s, c_s = x_screen[order], y_screen[order], colors[order]

    offsets, bf = _BLOCK_OFFSETS, _BLOCK_BF
    n_v, n_o = len(x_s), len(offsets)

    x_all = np.repeat(x_s, n_o) + np.tile(offsets[:, 0], n_v)
    y_all = np.repeat(y_s, n_o) + np.tile(offsets[:, 1], n_v)
    c_all = np.clip(
        np.repeat(c_s, n_o, axis=0).astype(np.float32) * np.tile(bf, n_v)[:, np.newaxis],
        0, 255).astype(np.uint8)

    valid = (x_all >= 0) & (x_all < occ_w) & (y_all >= 0) & (y_all < occ_h)
    canvas = np.full((occ_h, occ_w, 3), BG_COLOR, dtype=np.uint8)
    canvas[y_all[valid], x_all[valid]] = c_all[valid]
    return canvas


def render_occ_panel(npz_path, label, elev_deg, occ_h, occ_w):
    if npz_path and os.path.exists(npz_path):
        grid = np.load(npz_path)['semantics']
    else:
        grid = np.ones((200, 200, 20), dtype=np.uint8) * 17

    img = render_chase_cam(grid, elev_deg, occ_h, occ_w)
    bar = np.zeros((36, occ_w, 3), dtype=np.uint8)
    cv2.putText(bar, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    return np.vstack([bar, img])


def main():
    parser = argparse.ArgumentParser(description='ELAN three-way comparison video')
    parser.add_argument('--scene', default='citystreet_sunny_day_2025-09-25-15-38-56')
    parser.add_argument('--data_root', default='/data2/t113c52027/occ_gt_v2/data/elan')
    parser.add_argument('--occ_root',
                        default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--out_dir',
                        default=os.path.join(os.path.dirname(__file__), 'video_out'))
    parser.add_argument('--fps',     type=int,   default=10)
    parser.add_argument('--elev',    type=float, default=28)
    parser.add_argument('--total_w', type=int,   default=1920)
    parser.add_argument('--cam_h',   type=int,   default=90)
    parser.add_argument('--occ_h',   type=int,   default=480)
    args = parser.parse_args()

    scene_occ  = os.path.join(args.occ_root, args.scene)
    dir_raw    = os.path.join(scene_occ, 'raw')
    dir_heur   = os.path.join(scene_occ, 'heuristic')
    dir_seg    = os.path.join(scene_occ, 'seg')
    cam_dir    = os.path.join(args.data_root, args.scene, 'image')

    labels = ['Raw (no vehicles)', 'Heuristic', 'Semantic seg']
    dirs   = [dir_raw, dir_heur, dir_seg]
    for d, l in zip(dirs, labels):
        print(f"  [{'OK' if os.path.isdir(d) else 'MISSING'}] {l}: {d}")

    frames = sorted([f for f in os.listdir(dir_raw)
                     if os.path.exists(os.path.join(dir_raw, f, 'labels.npz'))])
    print(f"\nFrames: {len(frames)}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.scene}_compare.mp4')

    occ_panel_w = args.total_w // 3
    bar_h       = 36
    occ_panel_h = args.occ_h + bar_h
    total_h     = args.cam_h + occ_panel_h
    print(f"Canvas: {args.total_w}x{total_h}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (args.total_w, total_h))

    for frame_id in tqdm(frames, desc='Rendering'):
        # --- Front camera (full width) ---
        cam_path = os.path.join(cam_dir, f'{frame_id}.jpg')
        if os.path.exists(cam_path):
            cam_img = cv2.imread(cam_path)
            cam_img = cv2.resize(cam_img, (args.total_w, args.cam_h))
        else:
            cam_img = np.zeros((args.cam_h, args.total_w, 3), dtype=np.uint8)
        cv2.putText(cam_img, 'Front Camera', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # --- Three occupancy panels ---
        panels = []
        for d, lbl in zip(dirs, labels):
            npz = os.path.join(d, frame_id, 'labels.npz')
            panels.append(render_occ_panel(npz, lbl, args.elev, args.occ_h, occ_panel_w))

        occ_row = np.hstack(panels)
        for col in range(1, 3):
            cv2.line(occ_row, (col * occ_panel_w, 0), (col * occ_panel_w, occ_panel_h),
                     (80, 80, 80), 2)

        cv2.putText(occ_row, f'frame: {frame_id}',
                    (args.total_w - 200, occ_panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        writer.write(np.vstack([cam_img, occ_row]))

    writer.release()
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
