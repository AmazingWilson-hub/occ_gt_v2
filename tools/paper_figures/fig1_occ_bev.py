#!/usr/bin/env python3
"""
Figure 1: Semantic occupancy BEV visualization (Occ3D 17-class color map).
Picks one representative frame and saves occ_bev_visualization.png.

Usage:
    python3 fig1_occ_bev.py \
        --pred_dir ../../occupancy/nuscenes/v3/output/gt_pose/scene-0061 \
        --out paper_out/occ_bev_visualization.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OCC3D_COLORS = np.array([
    [20,  20,  20],        # 0: others
    [112, 128, 144],       # 1: barrier
    [220,  20,  60],       # 2: bicycle
    [255, 158,   0],       # 3: bus
    [255, 158,   0],       # 4: car
    [233, 150,  70],       # 5: construction_vehicle
    [255, 127,  80],       # 6: motorcycle
    [  0,   0, 230],       # 7: pedestrian
    [255, 215,   0],       # 8: traffic_cone
    [255, 140,   0],       # 9: trailer
    [255, 100,   0],       # 10: truck
    [  0, 207, 191],       # 11: driveable_surface
    [135, 206, 235],       # 12: other_flat
    [ 75,   0,  75],       # 13: sidewalk
    [  0, 175, 100],       # 14: terrain
    [222, 184, 135],       # 15: manmade
    [  0, 175,   0],       # 16: vegetation
    [  0,   0,   0],       # 17: free (black background)
], dtype=np.uint8)

CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]


def npz_to_bev_rgb(npz_path):
    data = np.load(npz_path)
    grid = data['semantics']          # (200, 200, 16)
    nx, ny, nz = grid.shape
    bev = np.full((nx, ny), 17, dtype=np.uint8)
    for z in range(nz):
        sl = grid[:, :, z]
        mask = sl != 17
        bev[mask] = sl[mask]
    rgb = OCC3D_COLORS[bev]           # (200, 200, 3)
    return bev, rgb


def pick_representative_token(pred_dir):
    tokens = sorted(os.listdir(pred_dir))
    mid = tokens[len(tokens) // 2]
    return mid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir',
                        default=os.path.join(os.path.dirname(__file__),
                                             '../../occupancy/nuscenes/v3/output/gt_pose/scene-0061'))
    parser.add_argument('--token', default=None,
                        help='Specific token to visualize. Defaults to middle frame.')
    parser.add_argument('--out', default='paper_out/occ_bev_visualization.png')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

    token = args.token or pick_representative_token(args.pred_dir)
    npz_path = os.path.join(args.pred_dir, token, 'labels.npz')
    print(f'Using token: {token}')
    print(f'Loading: {npz_path}')

    bev, rgb = npz_to_bev_rgb(npz_path)

    present = np.unique(bev)
    present = present[present != 17]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
    ax.axis('off')

    patches = [
        mpatches.Patch(color=OCC3D_COLORS[c] / 255.0, label=CLASS_NAMES[c])
        for c in present if c < 17
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=6,
              framealpha=0.7, ncol=2)

    fig.tight_layout(pad=0)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
