#!/usr/bin/env python3
"""
Render presentation-quality before/after figures for lane line accumulation.

Before: single frame ego-frame lane points + LiDAR
After:  N-frame accumulated lane points + LiDAR in world frame

Usage:
    python3 tools/paper_figures/render_lane_lines.py \
        --scene highway_sunny_day_2026-04-20-12-58-47 \
        --ref_frame 000020 \
        --window 60 \
        --out_dir paper_out/lanes
"""

import os
import sys
import json
import glob
import pickle
import argparse
import numpy as np

REPO = os.path.join(os.path.dirname(__file__), '..', '..')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pose_dict(scene):
    pkl = os.path.join(REPO, 'occupancy', 'g6', 'cvpr_format_occ_gen_g6',
                       'output', scene, 'pose_dict.pkl')
    if not os.path.exists(pkl):
        raise FileNotFoundError(f'pose_dict.pkl not found: {pkl}\nRun generate.py first.')
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def load_lane_pts_0429(json_path):
    with open(json_path) as f:
        d = json.load(f)
    xyz = d.get('xyz', [[], [], []])
    if not xyz[0]:
        return np.zeros((0, 3))
    lateral = np.array(xyz[0], dtype=np.float64)
    forward = np.array(xyz[1], dtype=np.float64)
    z       = np.array(xyz[2], dtype=np.float64)
    return np.stack([forward, lateral, z], axis=1)   # [x_fwd, y_lat, z]


def load_lidar_pts(pcd_path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pcd.points, dtype=np.float64)


def transform_pts(T, pts):
    if not len(pts):
        return pts
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def bev_figure(lidar_pts, lane_pts,
               title, out_path,
               range_x=(-60, 60), range_y=(-20, 20),
               lidar_color='#b0b8c1', lane_color='#FF6B00',
               figw=12, figh=5,
               lidar_s=0.3, lane_s=6,
               lidar_alpha=0.35, lane_alpha=0.9):
    """
    BEV plot: x = forward (vehicle), y = lateral.
    White background, minimal decoration.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(figw, figh), facecolor='white')
    ax.set_facecolor('white')

    # LiDAR background
    if len(lidar_pts):
        step = max(1, len(lidar_pts) // 150_000)
        lx = lidar_pts[::step, 0]
        ly = lidar_pts[::step, 1]
        mask = ((lx >= range_x[0]) & (lx <= range_x[1]) &
                (ly >= range_y[0]) & (ly <= range_y[1]))
        ax.scatter(lx[mask], ly[mask], s=lidar_s, c=lidar_color,
                   alpha=lidar_alpha, linewidths=0, rasterized=True)

    # Lane lines
    if len(lane_pts):
        lx = lane_pts[:, 0]
        ly = lane_pts[:, 1]
        mask = ((lx >= range_x[0]) & (lx <= range_x[1]) &
                (ly >= range_y[0]) & (ly <= range_y[1]))
        ax.scatter(lx[mask], ly[mask], s=lane_s, c=lane_color,
                   alpha=lane_alpha, linewidths=0, zorder=5, rasterized=True)

    # Ego car marker
    ax.plot(0, 0, marker='D', color='#2563EB', markersize=10, zorder=10,
            markeredgecolor='white', markeredgewidth=1.5)

    ax.set_xlim(range_x)
    ax.set_ylim(range_y)
    ax.set_aspect('equal')
    ax.set_xlabel('Forward (m)', fontsize=12)
    ax.set_ylabel('Lateral (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def side_by_side_figure(lidar_before, lane_before,
                        lidar_after,  lane_after,
                        out_path,
                        range_x=(-60, 60), range_y=(-20, 20)):
    """Single wide figure with Before | After side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(22, 5), facecolor='white')

    datasets = [
        (axes[0], lidar_before, lane_before, 'Before: Single Frame'),
        (axes[1], lidar_after,  lane_after,  'After: Multi-Frame Accumulation'),
    ]

    for ax, lidar_pts, lane_pts, title in datasets:
        ax.set_facecolor('white')

        if len(lidar_pts):
            step = max(1, len(lidar_pts) // 150_000)
            lx, ly = lidar_pts[::step, 0], lidar_pts[::step, 1]
            mask = ((lx >= range_x[0]) & (lx <= range_x[1]) &
                    (ly >= range_y[0]) & (ly <= range_y[1]))
            ax.scatter(lx[mask], ly[mask], s=0.3, c='#b0b8c1',
                       alpha=0.35, linewidths=0, rasterized=True)

        if len(lane_pts):
            lx, ly = lane_pts[:, 0], lane_pts[:, 1]
            mask = ((lx >= range_x[0]) & (lx <= range_x[1]) &
                    (ly >= range_y[0]) & (ly <= range_y[1]))
            ax.scatter(lx[mask], ly[mask], s=6, c='#FF6B00',
                       alpha=0.9, linewidths=0, zorder=5, rasterized=True)

        ax.plot(0, 0, marker='D', color='#2563EB', markersize=10, zorder=10,
                markeredgecolor='white', markeredgewidth=1.5)

        ax.set_xlim(range_x)
        ax.set_ylim(range_y)
        ax.set_aspect('equal')
        ax.set_xlabel('Forward (m)', fontsize=12)
        ax.set_ylabel('Lateral (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=10)

    # Shared legend
    handles = [
        mpatches.Patch(color='#b0b8c1', label='LiDAR points'),
        mpatches.Patch(color='#FF6B00', label='Lane line points'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#2563EB',
                   markersize=10, label='Ego vehicle'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene',     default='highway_sunny_day_2026-04-20-12-58-47')
    parser.add_argument('--ref_frame', default=None,
                        help='Reference frame ID, e.g. 000020 (default: middle of scene)')
    parser.add_argument('--window',    type=int, default=60,
                        help='Number of frames to accumulate for the "after" figure')
    parser.add_argument('--range_x',   type=float, nargs=2, default=[-80, 80],
                        help='Forward range in metres (default: -80 80)')
    parser.add_argument('--range_y',   type=float, nargs=2, default=[-15, 15],
                        help='Lateral range in metres (default: -15 15)')
    parser.add_argument('--out_dir',   default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(REPO, 'tools', 'paper_figures',
                                            'paper_out', 'lanes', args.scene)
    os.makedirs(out_dir, exist_ok=True)

    # Paths
    scene_dir = os.path.join(REPO, 'data', 'roadlane', '0429', args.scene)
    lane_dir  = os.path.join(scene_dir, 'lane')
    lidar_dir = os.path.join(scene_dir, 'VLS128_pcd')

    pose_dict = load_pose_dict(args.scene)
    frame_ids = sorted(pose_dict.keys())
    print(f'Scene: {args.scene}')
    print(f'Total frames with pose: {len(frame_ids)}')

    # Reference frame
    ref_fid = args.ref_frame or frame_ids[len(frame_ids) // 2]
    if ref_fid not in pose_dict:
        ref_fid = min(frame_ids, key=lambda f: abs(int(f) - int(ref_fid)))
    print(f'Reference frame: {ref_fid}')

    T_ref     = pose_dict[ref_fid]['matrix']          # world ← ref
    T_ref_inv = np.linalg.inv(T_ref)                  # ref ← world

    # ── BEFORE: single frame in ego frame ────────────────────────────────────
    print('\n[Before] Single frame ...')
    lidar_path = os.path.join(lidar_dir, f'{ref_fid}.pcd')
    lane_path  = os.path.join(lane_dir,  f'{ref_fid}.json')

    lidar_before = load_lidar_pts(lidar_path) if os.path.exists(lidar_path) else np.zeros((0,3))
    lane_before  = load_lane_pts_0429(lane_path) if os.path.exists(lane_path) else np.zeros((0,3))
    print(f'  LiDAR: {len(lidar_before):,} pts  |  Lane: {len(lane_before):,} pts')

    bev_figure(lidar_before, lane_before,
               title='Before: Single Frame',
               out_path=os.path.join(out_dir, 'before_single_frame.png'),
               range_x=args.range_x, range_y=args.range_y)

    # ── AFTER: window of frames accumulated, shown in ref frame coords ───────
    print('\n[After] Multi-frame accumulation ...')
    ref_idx   = frame_ids.index(ref_fid)
    half      = args.window // 2
    i_start   = max(0, ref_idx - half)
    i_end     = min(len(frame_ids), ref_idx + half)
    acc_ids   = frame_ids[i_start:i_end]
    print(f'  Accumulating frames {acc_ids[0]} → {acc_ids[-1]} ({len(acc_ids)} frames)')

    lidar_acc_list = []
    lane_acc_list  = []

    for fid in acc_ids:
        T = pose_dict[fid]['matrix']
        T_ego = T_ref_inv @ T   # world→ref → this frame in ref coords

        lidar_path = os.path.join(lidar_dir, f'{fid}.pcd')
        if os.path.exists(lidar_path):
            pts = load_lidar_pts(lidar_path)
            if len(pts):
                lidar_acc_list.append(transform_pts(T_ego, pts))

        lane_path = os.path.join(lane_dir, f'{fid}.json')
        if os.path.exists(lane_path):
            pts = load_lane_pts_0429(lane_path)
            if len(pts):
                lane_acc_list.append(transform_pts(T_ego, pts))

    lidar_after = np.vstack(lidar_acc_list) if lidar_acc_list else np.zeros((0,3))
    lane_after  = np.vstack(lane_acc_list)  if lane_acc_list  else np.zeros((0,3))
    print(f'  LiDAR: {len(lidar_after):,} pts  |  Lane: {len(lane_after):,} pts')

    bev_figure(lidar_after, lane_after,
               title='After: Multi-Frame Accumulation',
               out_path=os.path.join(out_dir, 'after_accumulated.png'),
               range_x=args.range_x, range_y=args.range_y)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print('\n[Side-by-side] ...')
    side_by_side_figure(lidar_before, lane_before,
                        lidar_after,  lane_after,
                        out_path=os.path.join(out_dir, 'before_after_comparison.png'),
                        range_x=args.range_x, range_y=args.range_y)

    print(f'\nAll figures saved to: {out_dir}')


if __name__ == '__main__':
    main()
