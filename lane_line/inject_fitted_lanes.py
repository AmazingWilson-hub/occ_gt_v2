#!/usr/bin/env python3
"""
Post-process existing occupancy npz files to inject fitted lane lines (label 18).

Uses world-frame fitted lane points from fit_lanes.py output,
transforms each frame's lane pts to ego frame, voxelizes, and overwrites
road (11) or free (17) voxels with label 18.

Usage:
    python3 lane_line/inject_fitted_lanes.py \
        --scene highway_sunny_day_2026-04-20-12-58-47 \
        --occ_dirs \
            occupancy/g6/cvpr_format_occ_gen_g6/output/highway_sunny_day_.../seg_main \
        --fitted_json lane_line/output/fitted/highway_.../fitted_lanes.json
"""

import os
import sys
import json
import pickle
import shutil
import argparse
import numpy as np
from tqdm import tqdm

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

GT_BOUNDS = np.array([-40.0, -40.0, -3.0, 40.0, 40.0, 5.4])
GT_VOXEL  = 0.4
GT_GRID   = (200, 200, 21)
LBL_ROAD  = 11
LBL_FREE  = 17
LBL_LANE  = 18
Z_LAYERS  = 3   # inject label in z, z+1, z+2 for thickness


def load_pose_dict(scene):
    pkl = os.path.join(REPO, 'occupancy', 'g6', 'cvpr_format_occ_gen_g6',
                       'output', scene, 'pose_dict.pkl')
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def load_fitted_lanes(json_path):
    """Load fitted lane points → list of Nx3 arrays in world frame [x_fwd, y_lat, z]."""
    with open(json_path) as f:
        lanes = json.load(f)
    result = []
    for lane in lanes:
        result.append(np.array(lane['points'], dtype=np.float64))
    return result


def transform_pts(T, pts):
    if not len(pts):
        return pts
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


def voxelize(pts_ego):
    """Convert ego-frame points to voxel indices. Returns Nx3 int array of valid indices."""
    ix = ((pts_ego[:, 0] - GT_BOUNDS[0]) / GT_VOXEL).astype(np.int32)
    iy = ((pts_ego[:, 1] - GT_BOUNDS[1]) / GT_VOXEL).astype(np.int32)
    iz = ((pts_ego[:, 2] - GT_BOUNDS[2]) / GT_VOXEL).astype(np.int32)
    valid = ((ix >= 0) & (ix < GT_GRID[0]) &
             (iy >= 0) & (iy < GT_GRID[1]) &
             (iz >= 0) & (iz < GT_GRID[2]))
    return np.stack([ix, iy, iz], axis=1)[valid]


def inject_lanes(occ, idxs):
    """Write LBL_LANE into occ where current label is road or free. Z-thickness = Z_LAYERS."""
    for dz in range(Z_LAYERS):
        iz = np.clip(idxs[:, 2] + dz, 0, GT_GRID[2] - 1)
        can = np.isin(occ[idxs[:, 0], idxs[:, 1], iz], [LBL_FREE, LBL_ROAD])
        occ[idxs[can, 0], idxs[can, 1], iz[can]] = LBL_LANE


def process_dir(occ_dir, out_dir, pose_dict, world_lane_pts):
    """Process one occupancy output directory (e.g. seg_main)."""
    os.makedirs(out_dir, exist_ok=True)
    frame_ids = sorted(f for f in os.listdir(occ_dir)
                       if os.path.exists(os.path.join(occ_dir, f, 'labels.npz')))
    injected = 0
    for fid in tqdm(frame_ids, desc=os.path.basename(occ_dir)):
        src_npz = os.path.join(occ_dir, fid, 'labels.npz')
        dst_dir = os.path.join(out_dir, fid)
        os.makedirs(dst_dir, exist_ok=True)
        dst_npz = os.path.join(dst_dir, 'labels.npz')

        occ = np.load(src_npz)['semantics'].copy()

        # Clear existing lane labels before injecting fitted lanes
        occ[occ == LBL_LANE] = LBL_FREE

        if fid in pose_dict:
            T_inv = np.linalg.inv(pose_dict[fid]['matrix'])
            for lane_w in world_lane_pts:
                pts_ego = transform_pts(T_inv, lane_w)
                idxs = voxelize(pts_ego)
                if len(idxs):
                    inject_lanes(occ, idxs)
            injected += 1

        np.savez_compressed(dst_npz, semantics=occ)

    print(f'  {injected}/{len(frame_ids)} frames had pose and received lane injection')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='highway_sunny_day_2026-04-20-12-58-47')
    parser.add_argument('--occ_dirs', nargs='+', default=None,
                        help='Occupancy output dirs to post-process. '
                             'Default: all subdirs under output/<scene>/')
    parser.add_argument('--fitted_json', default=None,
                        help='Path to fitted_lanes.json. '
                             'Default: lane_line/output/fitted/<scene>/fitted_lanes.json')
    parser.add_argument('--out_suffix', default='_fitted',
                        help='Suffix appended to each dir name for output '
                             '(default: _fitted, e.g. seg_main → seg_main_fitted)')
    args = parser.parse_args()

    # Defaults
    scene_out = os.path.join(REPO, 'occupancy', 'g6', 'cvpr_format_occ_gen_g6',
                             'output', args.scene)
    if args.occ_dirs is None:
        args.occ_dirs = [os.path.join(scene_out, d)
                         for d in os.listdir(scene_out)
                         if os.path.isdir(os.path.join(scene_out, d))
                         and d not in ('pose_dict.pkl',)
                         and not d.endswith('_fitted')]

    if args.fitted_json is None:
        args.fitted_json = os.path.join(REPO, 'lane_line', 'output', 'fitted',
                                        args.scene, 'fitted_lanes.json')

    print(f'Scene       : {args.scene}')
    print(f'Fitted JSON : {args.fitted_json}')
    print(f'Occ dirs    : {[os.path.basename(d) for d in args.occ_dirs]}')

    pose_dict       = load_pose_dict(args.scene)
    world_lane_pts  = load_fitted_lanes(args.fitted_json)
    print(f'Loaded {len(world_lane_pts)} fitted lanes, '
          f'{sum(len(l) for l in world_lane_pts)} total pts')

    for occ_dir in args.occ_dirs:
        out_dir = occ_dir + args.out_suffix
        print(f'\n[{os.path.basename(occ_dir)} → {os.path.basename(out_dir)}]')
        process_dir(occ_dir, out_dir, pose_dict, world_lane_pts)

    print('\nDone.')


if __name__ == '__main__':
    main()
