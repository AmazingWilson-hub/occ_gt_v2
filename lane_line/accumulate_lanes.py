"""
Lane line accumulation via multi-frame fusion.

Accumulates all LiDAR frames + lane line annotations into world frame
using KISS-ICP ego-poses, then saves a combined PLY and BEV image.

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

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
G6_POSE_DIR = os.path.join(SCRIPT_DIR, '..', 'occupancy', 'g6', 'cvpr_format_occ_gen_g6', 'pose_backends')
LANE_DATA   = os.path.join(SCRIPT_DIR, '..', 'data', 'roadlane', '0413')
G6_DATA     = os.path.join(SCRIPT_DIR, '..', 'data', 'g6')

sys.path.insert(0, G6_POSE_DIR)


def load_pose_dict(scene_name):
    from kiss_icp_gps import get_pose_dict
    scene_dir = os.path.join(G6_DATA, scene_name)
    print(f'Running KISS-ICP on {scene_dir} ...')
    return get_pose_dict(scene_dir)


def read_pcd_xyz(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float64)


def load_frame_lanes(json_path):
    """
    Returns list of (track_id, Nx3 pts) in vehicle frame [x-forward, y-left, z-up].
    JSON stores xyz as [y_lateral, x_forward, z_height].
    """
    with open(json_path) as f:
        d = json.load(f)
    lanes = []
    for lane in d['lane_lines']:
        y_vals = np.array(lane['xyz'][0], dtype=np.float64)
        x_vals = np.array(lane['xyz'][1], dtype=np.float64)
        z_vals = np.array(lane['xyz'][2], dtype=np.float64)
        pts = np.stack([x_vals, y_vals, z_vals], axis=1)
        lanes.append((lane['track_id'], pts))
    return lanes


def transform_pts(T, pts):
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


def accumulate(scene_name, out_dir, lidar_voxel=0.2):
    pose_dict = load_pose_dict(scene_name)

    pcd_dir  = os.path.join(G6_DATA, scene_name, 'VLS128_pcd')
    lane_dir = os.path.join(LANE_DATA, scene_name)

    lidar_pts_list = []
    lane_pts_list  = []
    lane_ids_list  = []

    frame_ids = sorted(pose_dict.keys())
    print(f'Accumulating {len(frame_ids)} frames ...')

    for fid in frame_ids:
        T = pose_dict[fid]['matrix']

        # LiDAR
        pcd_path = os.path.join(pcd_dir, f'{fid}.pcd')
        if os.path.exists(pcd_path):
            pts = read_pcd_xyz(pcd_path)
            if len(pts):
                lidar_pts_list.append(transform_pts(T, pts))

        # Lane lines
        json_path = os.path.join(lane_dir, f'{fid}.json')
        if os.path.exists(json_path):
            for track_id, pts in load_frame_lanes(json_path):
                world_pts = transform_pts(T, pts)
                lane_pts_list.append(world_pts)
                lane_ids_list.extend([track_id] * len(world_pts))

    lidar_pts = np.vstack(lidar_pts_list) if lidar_pts_list else np.zeros((0, 3))
    lane_pts  = np.vstack(lane_pts_list)  if lane_pts_list  else np.zeros((0, 3))
    lane_ids  = np.array(lane_ids_list)

    print(f'LiDAR: {len(lidar_pts):,} pts  |  Lane: {len(lane_pts):,} pts')

    os.makedirs(out_dir, exist_ok=True)
    _save_combined_ply(lidar_pts, lane_pts, lane_ids, lidar_voxel,
                       os.path.join(out_dir, 'scene_with_lanes.ply'))
    _save_bev(lidar_pts, lane_pts, lane_ids,
              os.path.join(out_dir, 'scene_with_lanes_bev.png'))
    print(f'Done. Saved to {out_dir}')


def _lane_colors(ids):
    unique_ids = np.unique(ids)
    rng = np.random.default_rng(42)
    color_map = {tid: rng.random(3) for tid in unique_ids}
    return np.array([color_map[i] for i in ids])


def _save_combined_ply(lidar_pts, lane_pts, lane_ids, voxel_size, path):
    pcd_lidar = o3d.geometry.PointCloud()
    pcd_lidar.points = o3d.utility.Vector3dVector(lidar_pts)
    pcd_lidar = pcd_lidar.voxel_down_sample(voxel_size)
    n_lidar = len(pcd_lidar.points)
    pcd_lidar.colors = o3d.utility.Vector3dVector(np.full((n_lidar, 3), 0.6))

    pcd_lane = o3d.geometry.PointCloud()
    pcd_lane.points = o3d.utility.Vector3dVector(lane_pts)
    pcd_lane.colors = o3d.utility.Vector3dVector(_lane_colors(lane_ids))

    combined = pcd_lidar + pcd_lane
    o3d.io.write_point_cloud(path, combined)
    print(f'  PLY ({n_lidar:,} lidar + {len(lane_pts):,} lane pts): {path}')


def _save_bev(lidar_pts, lane_pts, lane_ids, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 14))

    # LiDAR background
    step = max(1, len(lidar_pts) // 200_000)
    ax.scatter(lidar_pts[::step, 0], lidar_pts[::step, 1],
               s=0.1, c='#888888', alpha=0.3, rasterized=True)

    # Lane lines
    unique_ids = np.unique(lane_ids)
    rng = np.random.default_rng(42)
    color_map = {tid: rng.random(3) for tid in unique_ids}
    for tid in unique_ids:
        mask = lane_ids == tid
        ax.scatter(lane_pts[mask, 0], lane_pts[mask, 1],
                   s=2.0, c=[color_map[tid]], label=f'lane {tid}', zorder=5)

    if len(lane_pts):
        margin = 20.0
        ax.set_xlim(lane_pts[:, 0].min() - margin, lane_pts[:, 0].max() + margin)
        ax.set_ylim(lane_pts[:, 1].min() - margin, lane_pts[:, 1].max() + margin)

    ax.set_aspect('equal')
    ax.set_xlabel('x (forward, m)')
    ax.set_ylabel('y (left, m)')
    ax.set_title('Accumulated LiDAR + Lane Lines (BEV)')
    ax.legend(markerscale=8, loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  BEV: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='citystreet_sunny_day_2026-03-09-10-47-58')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--lidar_voxel', type=float, default=0.2,
                        help='Voxel size for LiDAR downsampling in PLY (default 0.2m)')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, 'output', args.scene)
    accumulate(args.scene, out_dir, args.lidar_voxel)
