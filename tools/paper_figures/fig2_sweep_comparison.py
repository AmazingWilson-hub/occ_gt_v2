#!/usr/bin/env python3
"""
Figure 2: Multi-frame accumulation comparison — 1 sweep (v1) vs 40 sweeps (v3).
Saves occ_sweep_1.png and occ_sweep_40.png side-by-side from the same token.

Usage:
    python3 fig2_sweep_comparison.py \
        --dir_1sweep  ../../occupancy/nuscenes/v1/output/scene-0061 \
        --dir_40sweep ../../occupancy/nuscenes/v3/output/gt_pose/scene-0061 \
        --out_dir paper_out
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

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
    [  0,   0,   0],       # 17: free (black)
], dtype=np.uint8)


def npz_to_bev_rgb(npz_path):
    data = np.load(npz_path)
    grid = data['semantics']
    nx, ny, nz = grid.shape
    bev = np.full((nx, ny), 17, dtype=np.uint8)
    for z in range(nz):
        sl = grid[:, :, z]
        mask = sl != 17
        bev[mask] = sl[mask]
    rgb = OCC3D_COLORS[bev]
    return rgb


def pick_common_token(dir_a, dir_b):
    tokens_a = set(os.listdir(dir_a))
    tokens_b = set(os.listdir(dir_b))
    common = sorted(tokens_a & tokens_b)
    assert common, 'No common tokens between the two directories.'
    return common[len(common) // 2]


def save_bev(rgb, path, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
    ax.set_title(title, fontsize=12, pad=4)
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_1sweep',
                        default=os.path.join(os.path.dirname(__file__),
                                             '../../occupancy/nuscenes/v1/output/scene-0061'))
    parser.add_argument('--dir_40sweep',
                        default=os.path.join(os.path.dirname(__file__),
                                             '../../occupancy/nuscenes/v3/output/gt_pose/scene-0061'))
    parser.add_argument('--token', default=None)
    parser.add_argument('--out_dir', default='paper_out')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    token = args.token or pick_common_token(args.dir_1sweep, args.dir_40sweep)
    print(f'Using token: {token}')

    path_1 = os.path.join(args.dir_1sweep,  token, 'labels.npz')
    path_40 = os.path.join(args.dir_40sweep, token, 'labels.npz')

    rgb_1  = npz_to_bev_rgb(path_1)
    rgb_40 = npz_to_bev_rgb(path_40)

    save_bev(rgb_1,  os.path.join(args.out_dir, 'occ_sweep_1.png'),  '1 sweep')
    save_bev(rgb_40, os.path.join(args.out_dir, 'occ_sweep_40.png'), '40 sweeps')

    # Also save a side-by-side combined figure for quick preview
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, rgb, title in zip(axes, [rgb_1, rgb_40], ['1 sweep', '40 sweeps']):
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
        ax.set_title(title, fontsize=13)
        ax.axis('off')
    fig.tight_layout()
    combined_path = os.path.join(args.out_dir, 'occ_sweep_comparison.png')
    fig.savefig(combined_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved combined: {combined_path}')


if __name__ == '__main__':
    main()
