"""
Inject lane lines into an existing occupancy npz as a new class (label 18).

Usage:
  python inject_lane.py --scene citystreet_sunny_day_2026-03-09-10-47-58 --frame 000012
"""

import os
import sys
import json
import argparse
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(SCRIPT_DIR, '..', 'data')
OCC_ROOT    = os.path.join(SCRIPT_DIR, '..', 'occupancy', 'g6', 'cvpr_format_occ_gen_g6', 'output')

# Occupancy grid parameters (g6)
GT_BOUNDS = np.array([-40.0, -40.0, -3.0, 40.0, 40.0, 5.4])
GT_VOXEL  = 0.4
GT_GRID   = (200, 200, 21)

LANE_LABEL = 18


def pts_to_voxel_idx(pts):
    """Convert Nx3 points (vehicle frame) to voxel indices. Returns (N,3) int array."""
    origin = GT_BOUNDS[:3]
    idx = ((pts - origin) / GT_VOXEL).astype(int)
    return idx


def load_lane_pts_0413(json_path):
    """0413 format: xyz[0]=y_lateral, xyz[1]=x_forward, xyz[2]=z."""
    with open(json_path) as f:
        d = json.load(f)
    all_pts = []
    for lane in d.get('lane_lines', []):
        y_vals = np.array(lane['xyz'][0], dtype=np.float64)
        x_vals = np.array(lane['xyz'][1], dtype=np.float64)
        z_vals = np.array(lane['xyz'][2], dtype=np.float64)
        all_pts.append(np.stack([x_vals, y_vals, z_vals], axis=1))
    if not all_pts:
        return np.zeros((0, 3))
    return np.vstack(all_pts)


def inject(scene, frame, out_dir):
    # Load occupancy
    npz_path = os.path.join(OCC_ROOT, scene, 'seg', frame, 'labels.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'No occupancy found: {npz_path}')
    npz = np.load(npz_path)
    grid = npz['semantics'].copy()
    print(f'Loaded occupancy {grid.shape}, labels: {np.unique(grid)}')

    # Load lane JSON (0413 format for citystreet)
    json_path = os.path.join(DATA_ROOT, 'roadlane', '0413', scene, f'{frame}.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'No lane JSON found: {json_path}')
    pts = load_lane_pts_0413(json_path)
    print(f'Lane points: {len(pts)}')

    if not len(pts):
        print('No lane points, nothing to inject.')
        return

    # Map points to voxel indices
    idx = pts_to_voxel_idx(pts)

    # Filter points inside grid bounds
    valid = (
        (idx[:, 0] >= 0) & (idx[:, 0] < GT_GRID[0]) &
        (idx[:, 1] >= 0) & (idx[:, 1] < GT_GRID[1]) &
        (idx[:, 2] >= 0) & (idx[:, 2] < GT_GRID[2])
    )
    idx = idx[valid]
    print(f'Valid voxels to fill: {len(idx)}')

    # Inject lane label
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = LANE_LABEL
    print(f'Labels after injection: {np.unique(grid)}')

    # Save
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'labels.npz')
    np.savez_compressed(out_path, semantics=grid)
    print(f'Saved: {out_path}')

    # Visualize BEV
    _save_bev(grid, os.path.join(out_dir, 'bev_with_lane.png'), frame)


def _save_bev(grid, path, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    label_colors = {
        2:  ('#90EE90', 'vegetation'),
        4:  ('#DC143C', 'car'),
        7:  ('#4B003F', 'pedestrian'),
        11: ('#D796F8', 'road'),
        15: ('#F7CE46', 'manmade'),
        17: ('#87CEEB', 'free'),
        18: ('#FF4500', 'lane line'),
    }

    # Collapse z: take max-label voxel per xy (ignoring free=17)
    bev = np.full((grid.shape[0], grid.shape[1]), 17, dtype=np.uint8)
    for iz in range(grid.shape[2]):
        layer = grid[:, :, iz]
        mask = layer != 17
        bev[mask] = layer[mask]

    # RGB image
    img = np.ones((*bev.shape, 3))
    for label, (color, _) in label_colors.items():
        rgb = tuple(int(color[i:i+2], 16) / 255 for i in (1, 3, 5))
        mask = bev == label
        img[mask] = rgb

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, origin='lower')
    ax.set_title(f'Occupancy BEV with Lane Lines — frame {title}')
    ax.set_xlabel('y voxel')
    ax.set_ylabel('x voxel')

    patches = [mpatches.Patch(color=c, label=n) for _, (c, n) in label_colors.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'BEV saved: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='citystreet_sunny_day_2026-03-09-10-47-58')
    parser.add_argument('--frame', default='000012')
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, 'output', args.scene, args.frame)
    inject(args.scene, args.frame, out_dir)
