#!/usr/bin/env python3
"""
Render occupancy grid: BEV (white background) + 3D voxel view via Open3D.

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
], dtype=np.uint8)

CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]

# Grid parameters (nuScenes v3/v4)
VOXEL_SIZE = 0.4   # metres
MIN_BOUND  = np.array([-40.0, -40.0, -1.0])


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


# ── 3D Open3D ────────────────────────────────────────────────────────────────


def render_3d(grid, out_path, elev=30, azim=-60, max_points=80000):
    """3D scatter plot using matplotlib. Works without GPU/EGL."""
    from mpl_toolkits.mplot3d import Axes3D

    xs, ys, zs = np.where(grid != 17)
    labels = grid[xs, ys, zs]

    # Convert voxel index → world coordinates (metres)
    cx = xs * VOXEL_SIZE + MIN_BOUND[0] + VOXEL_SIZE / 2.0
    cy = ys * VOXEL_SIZE + MIN_BOUND[1] + VOXEL_SIZE / 2.0
    cz = zs * VOXEL_SIZE + MIN_BOUND[2] + VOXEL_SIZE / 2.0

    colors = OCC3D_COLORS[labels].astype(np.float32) / 255.0

    # Subsample if too many points (for speed)
    n = len(cx)
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        cx, cy, cz, colors = cx[idx], cy[idx], cz[idx], colors[idx]

    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    ax.scatter(cx, cy, cz, c=colors, s=1.5, linewidths=0, alpha=0.85)

    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Z (m)', fontsize=9)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-1, 5.4)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 0.2])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
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
