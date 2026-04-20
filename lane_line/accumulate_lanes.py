"""
Lane line accumulation via multi-frame fusion.
Each frame's lane xyz (vehicle frame) is transformed to world frame
using KISS-ICP ego-poses, then accumulated into a global point cloud.

Usage:
  python accumulate_lanes.py --scene citystreet_sunny_day_2026-03-09-10-47-58
  python accumulate_lanes.py --scene citystreet_sunny_day_2026-04-08-16-41-27
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import open3d as o3d

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
G6_POSE_DIR  = os.path.join(SCRIPT_DIR, '..', 'occupancy', 'g6', 'cvpr_format_occ_gen_g6', 'pose_backends')
LANE_DATA    = os.path.join(SCRIPT_DIR, '..', 'data', 'roadlane', '0413')
G6_DATA      = os.path.join(SCRIPT_DIR, '..', 'data', 'g6')

sys.path.insert(0, G6_POSE_DIR)


def load_pose_dict(scene_name):
    from kiss_icp_gps import get_pose_dict
    scene_dir = os.path.join(G6_DATA, scene_name)
    print(f'Running KISS-ICP on {scene_dir} ...')
    return get_pose_dict(scene_dir)


def load_frame_lanes(json_path):
    """
    Returns list of Nx3 arrays in vehicle frame (x-forward, y-left, z-up).
    JSON stores xyz as [y_lateral, x_forward, z_height].
    """
    with open(json_path) as f:
        d = json.load(f)
    lanes = []
    for lane in d['lane_lines']:
        y_vals = np.array(lane['xyz'][0], dtype=np.float64)
        x_vals = np.array(lane['xyz'][1], dtype=np.float64)
        z_vals = np.array(lane['xyz'][2], dtype=np.float64)
        pts = np.stack([x_vals, y_vals, z_vals], axis=1)  # Nx3 [x,y,z]
        lanes.append((lane['track_id'], pts))
    return lanes


def accumulate(scene_name, out_dir):
    pose_dict = load_pose_dict(scene_name)

    lane_scene_dir = os.path.join(LANE_DATA, scene_name)
    json_files = sorted(glob.glob(os.path.join(lane_scene_dir, '*.json')))

    all_pts   = []
    all_ids   = []

    for jf in json_files:
        frame_id = os.path.splitext(os.path.basename(jf))[0]
        if frame_id not in pose_dict:
            continue

        T = pose_dict[frame_id]['matrix']  # 4x4 world-from-vehicle
        lanes = load_frame_lanes(jf)

        for track_id, pts in lanes:
            # pts: Nx3 vehicle frame → homogeneous Nx4
            ones = np.ones((len(pts), 1))
            pts_h = np.hstack([pts, ones])           # Nx4
            world_pts = (T @ pts_h.T).T[:, :3]       # Nx3 world frame
            all_pts.append(world_pts)
            all_ids.extend([track_id] * len(world_pts))

    if not all_pts:
        print('No lane points accumulated.')
        return

    all_pts = np.vstack(all_pts)
    all_ids = np.array(all_ids)
    print(f'Accumulated {len(all_pts)} points from {len(json_files)} frames.')

    os.makedirs(out_dir, exist_ok=True)
    _save_ply(all_pts, all_ids, os.path.join(out_dir, 'lanes_world.ply'))
    _save_bev(all_pts, all_ids, os.path.join(out_dir, 'lanes_bev.png'))
    print(f'Saved to {out_dir}')


def _save_ply(pts, ids, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # Colour by track_id
    unique_ids = np.unique(ids)
    rng = np.random.default_rng(42)
    color_map = {tid: rng.random(3) for tid in unique_ids}
    colors = np.array([color_map[i] for i in ids])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)
    print(f'  PLY: {path}')


def _save_bev(pts, ids, path, resolution=0.1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x, y = pts[:, 0], pts[:, 1]
    margin = 5.0
    x_min, x_max = x.min() - margin, x.max() + margin
    y_min, y_max = y.min() - margin, y.max() + margin

    unique_ids = np.unique(ids)
    rng = np.random.default_rng(42)
    color_map = {tid: rng.random(3) for tid in unique_ids}

    fig, ax = plt.subplots(figsize=(12, 12))
    for tid in unique_ids:
        mask = ids == tid
        ax.scatter(x[mask], y[mask], s=0.5, c=[color_map[tid]], label=f'lane {tid}')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('x (forward)')
    ax.set_ylabel('y (left)')
    ax.set_title('Accumulated Lane Lines (BEV)')
    ax.legend(markerscale=10, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  BEV: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='citystreet_sunny_day_2026-03-09-10-47-58')
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, 'output', args.scene)
    accumulate(args.scene, out_dir)
