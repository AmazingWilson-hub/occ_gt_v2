#!/usr/bin/env python3
"""
Figure 3: Pose backend comparison — GT pose vs KISS-SLAM (both 40 sweeps, v3).
Saves occ_pose_gt.png and occ_pose_kiss.png from the same token.

Usage:
    python3 fig3_pose_comparison.py \
        --dir_gt   ../../occupancy/nuscenes/v3/output/gt_pose/scene-0061 \
        --dir_kiss ../../occupancy/nuscenes/v3/output/kiss_slam/scene-0061 \
        --out_dir  paper_out
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
    parser.add_argument('--dir_gt',
                        default=os.path.join(os.path.dirname(__file__),
                                             '../../occupancy/nuscenes/v3/output/gt_pose/scene-0061'))
    parser.add_argument('--dir_kiss',
                        default=os.path.join(os.path.dirname(__file__),
                                             '../../occupancy/nuscenes/v3/output/kiss_slam/scene-0061'))
    parser.add_argument('--token', default=None)
    parser.add_argument('--out_dir', default='paper_out')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    token = args.token or pick_common_token(args.dir_gt, args.dir_kiss)
    print(f'Using token: {token}')

    path_gt   = os.path.join(args.dir_gt,   token, 'labels.npz')
    path_kiss = os.path.join(args.dir_kiss, token, 'labels.npz')

    rgb_gt   = npz_to_bev_rgb(path_gt)
    rgb_kiss = npz_to_bev_rgb(path_kiss)

    save_bev(rgb_gt,   os.path.join(args.out_dir, 'occ_pose_gt.png'),   'GT pose')
    save_bev(rgb_kiss, os.path.join(args.out_dir, 'occ_pose_kiss.png'), 'KISS-SLAM pose')

    # Also save combined preview
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, rgb, title in zip(axes, [rgb_gt, rgb_kiss], ['GT pose', 'KISS-SLAM pose']):
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower')
        ax.set_title(title, fontsize=13)
        ax.axis('off')
    fig.tight_layout()
    combined_path = os.path.join(args.out_dir, 'occ_pose_comparison.png')
    fig.savefig(combined_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved combined: {combined_path}')


if __name__ == '__main__':
    main()
