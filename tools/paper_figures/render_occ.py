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


def render_3d(grid, out_path, elev_deg=28, occ_w=1200, occ_h=700,
              view_range=35.0, z_offset=-2.0, voxel_style='flat'):
    """
    Chase-cam style 3D render, same projection as tools/compare_video/renderer.py.
    White background, RGB output.
    """
    occupied = (grid != 17) & (grid != 0)
    xs, ys, zs = np.where(occupied)
    px = (xs - 100.0) * VOXEL_SIZE
    py = (ys - 100.0) * VOXEL_SIZE
    pz = zs * VOXEL_SIZE + z_offset

    labels = grid[xs, ys, zs]
    colors = OCC3D_COLORS[labels].astype(np.float32)

    import cv2

    elev = np.radians(elev_deg)
    se, ce = np.sin(elev), np.cos(elev)
    scale = occ_w / (2.0 * view_range)
    B = max(3, int(np.ceil(VOXEL_SIZE * scale)))   # block size in pixels

    # Project voxel bottom-left corner to screen
    # Each voxel (xi, yi, zi): world offset per voxel unit:
    #   +x → screen (0,       -se*scale*V)
    #   +y → screen (-scale*V, 0         )
    #   +z → screen (0,       -ce*scale*V)
    V = VOXEL_SIZE
    dx_x, dy_x = 0,            int(se * scale * V)   # +x moves up
    dx_y, dy_y = int(scale*V), 0                     # +y moves right
    dx_z, dy_z = 0,            int(ce * scale * V)   # +z moves up

    ego_sx = occ_w // 2
    ego_sy = int(occ_h * 0.68)

    def world_to_screen(px_, py_, pz_):
        sx = int(-py_ * scale + ego_sx)
        sy = int(ego_sy - (px_ * se + pz_ * ce) * scale)
        return sx, sy

    canvas = np.full((occ_h, occ_w, 3), 255, dtype=np.uint8)

    # sort back-to-front for painter's algorithm
    x_2d  = -py
    y_2d  =  px * se + pz * ce
    depth = -px * ce + pz * se
    order = np.argsort(depth)

    colors_base = OCC3D_COLORS[labels]

    for idx in order:
        sx, sy = world_to_screen(px[idx], py[idx], pz[idx])

        c = colors_base[idx].astype(np.float32)
        c_top   = np.clip(c * 1.00, 0, 255).astype(np.uint8)
        c_front = np.clip(c * 0.65, 0, 255).astype(np.uint8)
        c_side  = np.clip(c * 0.45, 0, 255).astype(np.uint8)

        # Top face: parallelogram spanned by +x and +y offsets
        top = np.array([
            [sx,              sy],
            [sx + dx_x,       sy - dy_x],
            [sx + dx_x+dx_y,  sy - dy_x - dy_y],
            [sx + dx_y,       sy - dy_y],
        ], dtype=np.int32)

        # Front face: spanned by +x and +z offsets (y face, viewer side)
        front = np.array([
            [sx + dx_y,           sy - dy_y],
            [sx + dx_x + dx_y,    sy - dy_x - dy_y],
            [sx + dx_x + dx_y,    sy - dy_x - dy_y - dy_z],
            [sx + dx_y,           sy - dy_y - dy_z],
        ], dtype=np.int32)

        # Side face: spanned by +y and +z offsets (x face, viewer side)
        side = np.array([
            [sx,              sy],
            [sx + dx_y,       sy - dy_y],
            [sx + dx_y,       sy - dy_y - dy_z],
            [sx,              sy - dy_z],
        ], dtype=np.int32)

        cv2.fillPoly(canvas, [side],  c_side.tolist())
        cv2.fillPoly(canvas, [front], c_front.tolist())
        cv2.fillPoly(canvas, [top],   c_top.tolist())

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
