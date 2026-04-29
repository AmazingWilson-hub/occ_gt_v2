"""
Lane line accumulation via multi-frame fusion.

Supports two dataset formats:
  - 0413 (citystreet): lane data in roadlane/0413/<scene>/, LiDAR in g6/<scene>/
    Lane JSON: {intrinsic, extrinsic, lane_lines: [{track_id, xyz:[y,x,z], uv}]}
  - 0429 (highway): self-contained in roadlane/0429/<scene>/
    Lane JSON: {xyz: [x_lateral, y_forward, z_height]} (flat, no per-lane grouping)

Usage:
  python accumulate_lanes.py --scene citystreet_sunny_day_2026-03-09-10-47-58
  python accumulate_lanes.py --scene citystreet_sunny_day_2026-04-08-16-41-27
  python accumulate_lanes.py --scene highway_sunny_day_2026-04-20-12-58-47
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import open3d as o3d

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(SCRIPT_DIR, '..', 'data')
G6_POSE_DIR = os.path.join(SCRIPT_DIR, '..', 'occupancy', 'g6', 'cvpr_format_occ_gen_g6', 'pose_backends')

sys.path.insert(0, G6_POSE_DIR)


# ---------------------------------------------------------------------------
# Dataset layout detection
# ---------------------------------------------------------------------------

def find_scene(scene_name):
    """Return (scene_dir, lane_dir, lidar_dir, batch) or raise."""
    for batch in sorted(os.listdir(os.path.join(DATA_ROOT, 'roadlane'))):
        candidate = os.path.join(DATA_ROOT, 'roadlane', batch, scene_name)
        if os.path.isdir(candidate):
            # 0429 style: LiDAR lives inside the scene dir
            lidar_inside = os.path.join(candidate, 'VLS128_pcd')
            lane_inside  = os.path.join(candidate, 'lane')
            if os.path.isdir(lidar_inside) and os.path.isdir(lane_inside):
                return candidate, lane_inside, lidar_inside, batch
            # 0413 style: LiDAR lives in g6/
            lidar_g6 = os.path.join(DATA_ROOT, 'g6', scene_name, 'VLS128_pcd')
            if os.path.isdir(lidar_g6):
                return candidate, candidate, lidar_g6, batch
    raise FileNotFoundError(f'Scene not found: {scene_name}')


# ---------------------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------------------

def load_pose_dict(scene_dir):
    from kiss_icp_gps import get_pose_dict
    print(f'Running KISS-ICP on {scene_dir} ...')
    return get_pose_dict(scene_dir)


# ---------------------------------------------------------------------------
# Lane loading (handles both formats)
# ---------------------------------------------------------------------------

def load_frame_lanes_0413(json_path):
    """
    0413 format: lane_lines list with track_id.
    JSON xyz order: [y_lateral, x_forward, z_height] → convert to [x,y,z].
    Returns list of (track_id, Nx3 array).
    """
    with open(json_path) as f:
        d = json.load(f)
    result = []
    for lane in d.get('lane_lines', []):
        y_vals = np.array(lane['xyz'][0], dtype=np.float64)
        x_vals = np.array(lane['xyz'][1], dtype=np.float64)
        z_vals = np.array(lane['xyz'][2], dtype=np.float64)
        pts = np.stack([x_vals, y_vals, z_vals], axis=1)
        result.append((lane['track_id'], pts))
    return result


def load_frame_lanes_0429(json_path):
    """
    0429 format: flat xyz, no per-lane grouping, track_id = 0.
    JSON xyz order: [x_lateral, y_forward, z_height].
    Returns list of (track_id, Nx3 array).
    """
    with open(json_path) as f:
        d = json.load(f)
    xyz = d.get('xyz', [[], [], []])
    if not xyz[0]:
        return []
    x_vals = np.array(xyz[0], dtype=np.float64)
    y_vals = np.array(xyz[1], dtype=np.float64)
    z_vals = np.array(xyz[2], dtype=np.float64)
    pts = np.stack([x_vals, y_vals, z_vals], axis=1)
    return [(0, pts)]


# ---------------------------------------------------------------------------
# Core accumulation
# ---------------------------------------------------------------------------

def transform_pts(T, pts):
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


def read_pcd_xyz(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float64)


def accumulate(scene_name, out_dir, lidar_voxel=0.2):
    scene_dir, lane_dir, lidar_dir, batch = find_scene(scene_name)
    print(f'Scene  : {scene_dir}')
    print(f'LiDAR  : {lidar_dir}')
    print(f'Lane   : {lane_dir}')
    print(f'Batch  : {batch}')

    load_lanes = load_frame_lanes_0429 if batch == '0429' else load_frame_lanes_0413

    pose_dict = load_pose_dict(scene_dir)

    lidar_pts_list = []
    lane_pts_list  = []
    lane_ids_list  = []

    frame_ids = sorted(pose_dict.keys())
    print(f'Accumulating {len(frame_ids)} frames ...')

    for fid in frame_ids:
        T = pose_dict[fid]['matrix']

        # LiDAR
        pcd_path = os.path.join(lidar_dir, f'{fid}.pcd')
        if os.path.exists(pcd_path):
            pts = read_pcd_xyz(pcd_path)
            if len(pts):
                lidar_pts_list.append(transform_pts(T, pts))

        # Lane lines
        json_path = os.path.join(lane_dir, f'{fid}.json')
        if os.path.exists(json_path):
            for track_id, pts in load_lanes(json_path):
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


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

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

    # LiDAR background (subsample for speed)
    step = max(1, len(lidar_pts) // 200_000)
    ax.scatter(lidar_pts[::step, 0], lidar_pts[::step, 1],
               s=0.1, c='#888888', alpha=0.3, rasterized=True)

    # Lane lines coloured by track_id
    unique_ids = np.unique(lane_ids)
    rng = np.random.default_rng(42)
    color_map = {tid: rng.random(3) for tid in unique_ids}
    for tid in unique_ids:
        mask = lane_ids == tid
        label = f'lane {tid}' if len(unique_ids) > 1 else 'lane'
        ax.scatter(lane_pts[mask, 0], lane_pts[mask, 1],
                   s=2.0, c=[color_map[tid]], label=label, zorder=5)

    if len(lane_pts):
        margin = 20.0
        ax.set_xlim(lane_pts[:, 0].min() - margin, lane_pts[:, 0].max() + margin)
        ax.set_ylim(lane_pts[:, 1].min() - margin, lane_pts[:, 1].max() + margin)

    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Accumulated LiDAR + Lane Lines (BEV)')
    ax.legend(markerscale=8, loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  BEV: {path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='highway_sunny_day_2026-04-20-12-58-47')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--lidar_voxel', type=float, default=0.2)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, 'output', args.scene)
    accumulate(args.scene, out_dir, args.lidar_voxel)
