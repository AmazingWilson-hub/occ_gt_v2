#!/usr/bin/env python3
"""
Render occupancy grid: BEV (white background) + 3D chase-cam view.

Usage:
    python3 render_occ.py \
        --npz /path/to/labels.npz \
        --out_dir paper_out
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Occ3D 17-class color map (RGB 0-255)
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
    [255, 255, 255],   # 17: free  <- white background
    [255, 255, 255],   # 18: lane line (white)
], dtype=np.uint8)

CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free', 'lane_line',
]

# Grid parameters (nuScenes v3/v4)
VOXEL_SIZE = 0.4   # metres
MIN_BOUND  = np.array([-40.0, -40.0, -1.0])
VIEW_RANGE = 35.0


def load_grid(npz_path):
    return np.load(npz_path)['semantics']   # (200, 200, 16) uint8


# ── BEV ──────────────────────────────────────────────────────────────────────

def render_bev(grid, out_path):
    nx, ny, nz = grid.shape
    bev = np.full((nx, ny), 17, dtype=np.uint8)
    for z in range(nz):
        sl = grid[:, :, z]
        mask = sl != 17
        bev[mask] = sl[mask]

    rgb = OCC3D_COLORS[bev]   # white where free

    present = np.unique(bev)
    present = present[present != 17]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
    ax.set_facecolor('white')
    ax.axis('off')

    patches = [
        mpatches.Patch(color=OCC3D_COLORS[c] / 255.0, label=CLASS_NAMES[c])
        for c in present if c < 17
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=6,
              framealpha=0.85, ncol=2)

    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'BEV saved: {out_path}')


# ── 3D chase-cam (same projection as tools/compare_video/renderer.py) ────────


def render_3d(grid, out_path, elev_deg=28, occ_w=1200, occ_h=700, z_offset=-2.0):
    """
    Chase-cam 3D render identical to tools/compare_video/renderer.py.
    x_2d = -py, y_2d = px*sin(el) + pz*cos(el), depth = -px*cos(el) + pz*sin(el).
    Block rendering with z-height brightness. White background.
    """
    import cv2

    occupied = (grid != 17) & (grid != 0)
    if not np.any(occupied):
        cv2.imwrite(out_path, np.full((occ_h, occ_w, 3), 255, dtype=np.uint8))
        return

    xs, ys, zs = np.where(occupied)
    px = (xs - 100.0) * VOXEL_SIZE
    py = (ys - 100.0) * VOXEL_SIZE
    pz = zs * VOXEL_SIZE + z_offset
    labels = grid[xs, ys, zs]

    elev  = np.radians(elev_deg)
    se, ce = np.sin(elev), np.cos(elev)

    scale      = occ_w / (2.0 * VIEW_RANGE)
    block_size = max(4, int(np.ceil(VOXEL_SIZE * scale)))

    # z-height brightness: same as renderer.py flat mode (0.6 + 0.4*z_norm)
    z_norm = (pz - pz.min()) / (pz.max() - pz.min() + 1e-6)
    colors = OCC3D_COLORS[labels].astype(np.float32)
    colors = np.clip(colors * (0.6 + 0.4 * z_norm)[:, np.newaxis], 0, 255).astype(np.uint8)

    # same projection as renderer.py
    x_2d  = -py
    y_2d  =  px * se + pz * ce
    depth = -px * ce + pz * se

    ego_sx = occ_w // 2
    ego_sy = int(occ_h * 0.70)
    x_scr  = (x_2d * scale + ego_sx).astype(np.int32)
    y_scr  = (ego_sy - y_2d * scale).astype(np.int32)

    margin  = block_size
    visible = ((x_scr >= -margin) & (x_scr < occ_w + margin) &
               (y_scr >= -margin) & (y_scr < occ_h + margin))
    x_scr, y_scr = x_scr[visible], y_scr[visible]
    colors, depth = colors[visible], depth[visible]

    order = np.argsort(depth)
    x_s, y_s, c_s = x_scr[order], y_scr[order], colors[order]

    # vectorised block paint
    B = block_size
    oy, ox = np.mgrid[0:B, 0:B]
    offsets = np.stack([ox.ravel(), oy.ravel()], axis=1)
    n_v, n_o = len(x_s), len(offsets)

    x_all = np.repeat(x_s, n_o) + np.tile(offsets[:, 0], n_v)
    y_all = np.repeat(y_s, n_o) + np.tile(offsets[:, 1], n_v)
    c_all = np.repeat(c_s, n_o, axis=0)

    valid  = (x_all >= 0) & (x_all < occ_w) & (y_all >= 0) & (y_all < occ_h)
    canvas = np.full((occ_h, occ_w, 3), 255, dtype=np.uint8)
    canvas[y_all[valid], x_all[valid]] = c_all[valid]

    cv2.imwrite(out_path, canvas[:, :, ::-1])
    print(f'3D saved: {out_path}')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',
        default='/home/t113c52027/t113c52027/occ_gt_v2/occupancy/nuscenes/v4'
                '/output/kiss_slam/scene-0061'
                '/88449a5cb1644a199c1c11f6ac034867/labels.npz')
    parser.add_argument('--out_dir', default='paper_out')
    parser.add_argument('--no_3d', action='store_true',
                        help='Skip 3D rendering (faster)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    grid = load_grid(args.npz)
    print(f'Grid shape: {grid.shape}')
    classes = np.unique(grid)
    print(f'Classes present: {classes[classes != 17].tolist()}')

    render_bev(grid, os.path.join(args.out_dir, 'occ_bev_visualization.png'))

    if not args.no_3d:
        render_3d(grid, os.path.join(args.out_dir, 'occ_3d_view.png'))


if __name__ == '__main__':
    main()
