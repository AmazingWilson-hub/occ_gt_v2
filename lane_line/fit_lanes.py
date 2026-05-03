#!/usr/bin/env python3
"""
Fit smooth lane lines to accumulated lane point cloud.

Pipeline:
  1. Re-accumulate lane points from 0429 JSON files using cached pose_dict
  2. Cluster by lateral (y) position → individual lanes
  3. Polynomial fit per lane in forward (x) direction
  4. Output clean lane points + comparison figure

Usage:
    python3 lane_line/fit_lanes.py \
        --scene highway_sunny_day_2026-04-20-12-58-47 \
        --out_dir lane_line/output/fitted
"""

import os
import sys
import json
import glob
import pickle
import argparse
import numpy as np
from scipy.signal import savgol_filter

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pose_dict(scene):
    pkl = os.path.join(REPO, 'occupancy', 'g6', 'cvpr_format_occ_gen_g6',
                       'output', scene, 'pose_dict.pkl')
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def load_lane_0429(json_path):
    with open(json_path) as f:
        d = json.load(f)
    xyz = d.get('xyz', [[], [], []])
    if not xyz[0]:
        return np.zeros((0, 3))
    lateral = np.array(xyz[0], dtype=np.float64)
    forward = np.array(xyz[1], dtype=np.float64)
    z       = np.array(xyz[2], dtype=np.float64)
    return np.stack([forward, lateral, z], axis=1)  # [x_fwd, y_lat, z]


def transform_pts(T, pts):
    if not len(pts):
        return pts
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


def accumulate_lanes(scene, max_frames=None):
    """Return accumulated lane points in world frame."""
    lane_dir  = os.path.join(REPO, 'data', 'roadlane', '0429', scene, 'lane')
    pose_dict = load_pose_dict(scene)
    frame_ids = sorted(pose_dict.keys())
    if max_frames:
        frame_ids = frame_ids[:max_frames]

    pts_list = []
    for fid in frame_ids:
        jp = os.path.join(lane_dir, f'{fid}.json')
        if not os.path.exists(jp):
            continue
        pts = load_lane_0429(jp)
        if not len(pts):
            continue
        T = pose_dict[fid]['matrix']
        pts_list.append(transform_pts(T, pts))

    if not pts_list:
        raise RuntimeError('No lane points found')
    return np.vstack(pts_list)


# ---------------------------------------------------------------------------
# Lane clustering
# ---------------------------------------------------------------------------

def cluster_by_lateral(pts, min_pts=30, n_bins=200, smooth_sigma=2.0):
    """
    Split pts into lanes by finding valleys in the lateral (y) histogram.
    Returns list of Nx3 arrays, one per lane.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    ys = pts[:, 1]
    y_min, y_max = ys.min(), ys.max()
    hist, edges = np.histogram(ys, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_w = centers[1] - centers[0]

    # Smooth histogram and find peaks (density modes = lane centers)
    smoothed = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma / bin_w)
    peaks, props = find_peaks(smoothed, height=min_pts * 0.5, distance=1.0 / bin_w)

    if len(peaks) == 0:
        # Fallback: single cluster
        return [pts] if len(pts) >= min_pts else []

    # Find valleys between consecutive peaks as cut points
    cut_ys = []
    for i in range(len(peaks) - 1):
        lo, hi = peaks[i], peaks[i + 1]
        valley_idx = lo + np.argmin(smoothed[lo:hi + 1])
        cut_ys.append(centers[valley_idx])

    # Split points at cut positions
    ys_sorted_idx = np.argsort(ys)
    ys_sorted = ys[ys_sorted_idx]
    split_pos = np.searchsorted(ys_sorted, cut_ys)
    groups = np.split(ys_sorted_idx, split_pos)

    lanes = [pts[g] for g in groups if len(g) >= min_pts]
    return lanes


# ---------------------------------------------------------------------------
# Per-lane polynomial fitting
# ---------------------------------------------------------------------------

def fit_lane(pts, degree=3, n_samples=300, smooth_window=21):
    """
    Fit a polynomial to one lane's points.
    Returns (x_clean, y_clean, z_clean) arrays.
    """
    x = pts[:, 0]   # forward
    y = pts[:, 1]   # lateral
    z = pts[:, 2]   # height

    x_range = (x.min(), x.max())
    x_span  = x_range[1] - x_range[0]
    if x_span < 5.0:
        return None

    # Polynomial fit y = f(x) and z = g(x)
    coeff_y = np.polyfit(x, y, degree)
    coeff_z = np.polyfit(x, z, degree)

    x_clean = np.linspace(x_range[0], x_range[1], n_samples)
    y_clean = np.polyval(coeff_y, x_clean)
    z_clean = np.polyval(coeff_z, x_clean)

    # Optional Savitzky-Golay smoothing on residuals
    if smooth_window and n_samples > smooth_window:
        y_clean = savgol_filter(y_clean, window_length=smooth_window, polyorder=2)
        z_clean = savgol_filter(z_clean, window_length=smooth_window, polyorder=2)

    return np.stack([x_clean, y_clean, z_clean], axis=1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_comparison(raw_pts, fitted_lanes, out_path,
                    range_x=(-200, 200), range_y=(-15, 15)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    LANE_COLORS = ['#E63946', '#2196F3', '#4CAF50', '#FF9800',
                   '#9C27B0', '#00BCD4', '#FF5722', '#8BC34A']

    fig, axes = plt.subplots(1, 2, figsize=(22, 6), facecolor='white')

    # ── Left: raw accumulated ─────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('white')
    ax.scatter(raw_pts[:, 0], raw_pts[:, 1],
               s=1.0, c='#FF6B00', alpha=0.4, linewidths=0, rasterized=True)
    ax.set_xlim(range_x); ax.set_ylim(range_y)
    ax.set_aspect('equal')
    ax.set_title('Accumulated (Raw)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Forward (m)'); ax.set_ylabel('Lateral (m)')
    ax.spines[['top', 'right']].set_visible(False)

    # ── Right: fitted lanes ───────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor('white')
    for i, lane_pts in enumerate(fitted_lanes):
        c = LANE_COLORS[i % len(LANE_COLORS)]
        ax.plot(lane_pts[:, 0], lane_pts[:, 1],
                color=c, linewidth=2.5, solid_capstyle='round', label=f'Lane {i+1}')
    ax.set_xlim(range_x); ax.set_ylim(range_y)
    ax.set_aspect('equal')
    ax.set_title('Fitted Lane Lines', fontsize=14, fontweight='bold')
    ax.set_xlabel('Forward (m)'); ax.set_ylabel('Lateral (m)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=10, loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Comparison: {out_path}')


def plot_fitted_only(fitted_lanes, out_path,
                     range_x=(-200, 200), range_y=(-15, 15)):
    """Clean figure of fitted lanes only — for presentation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    LANE_COLORS = ['#E63946', '#2196F3', '#4CAF50', '#FF9800',
                   '#9C27B0', '#00BCD4', '#FF5722', '#8BC34A']

    fig, ax = plt.subplots(figsize=(16, 5), facecolor='white')
    ax.set_facecolor('white')

    for i, lane_pts in enumerate(fitted_lanes):
        c = LANE_COLORS[i % len(LANE_COLORS)]
        ax.plot(lane_pts[:, 0], lane_pts[:, 1],
                color=c, linewidth=3, solid_capstyle='round', label=f'Lane {i+1}')

    ax.set_xlim(range_x); ax.set_ylim(range_y)
    ax.set_aspect('equal')
    ax.set_title('Fitted Lane Lines (World Frame)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Forward (m)'); ax.set_ylabel('Lateral (m)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=11, loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Fitted only: {out_path}')


def save_fitted_json(fitted_lanes, out_path):
    """Save fitted lanes in the same xyz format as original 0429 JSON."""
    result = []
    for i, pts in enumerate(fitted_lanes):
        # pts: Nx3 [x_fwd, y_lat, z]
        result.append({
            'lane_id': i,
            'xyz': [pts[:, 1].tolist(),   # lateral (matches 0429 format)
                    pts[:, 0].tolist(),   # forward
                    pts[:, 2].tolist()],  # z
            'n_pts': len(pts),
        })
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  JSON: {out_path}  ({len(result)} lanes)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene',         default='highway_sunny_day_2026-04-20-12-58-47')
    parser.add_argument('--degree',        type=int,   default=4,
                        help='Polynomial degree for lane fitting (default: 4)')
    parser.add_argument('--gap',           type=float, default=1.5,
                        help='Lateral gap threshold to split lanes (m, default: 1.5)')
    parser.add_argument('--min_pts',       type=int,   default=50,
                        help='Minimum points per lane cluster (default: 50)')
    parser.add_argument('--n_samples',     type=int,   default=500,
                        help='Sample points per fitted lane (default: 500)')
    parser.add_argument('--range_x',       type=float, nargs=2, default=[-200, 200])
    parser.add_argument('--range_y',       type=float, nargs=2, default=[-15,  15])
    parser.add_argument('--out_dir',       default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        REPO, 'lane_line', 'output', 'fitted', args.scene)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Accumulate
    print(f'[1/3] Accumulating lane points for {args.scene} ...')
    raw_pts = accumulate_lanes(args.scene)
    print(f'      Total raw points: {len(raw_pts):,}')

    # 2. Cluster into individual lanes
    print(f'[2/3] Clustering into lanes (min_pts={args.min_pts}) ...')
    lane_clusters = cluster_by_lateral(raw_pts, min_pts=args.min_pts)
    print(f'      Found {len(lane_clusters)} lane(s):')
    for i, lc in enumerate(lane_clusters):
        y_mean = lc[:, 1].mean()
        print(f'        Lane {i+1}: {len(lc):,} pts, mean lateral={y_mean:.1f}m')

    # 3. Fit
    print(f'[3/3] Fitting polynomials (degree={args.degree}) ...')
    fitted_lanes = []
    for i, lc in enumerate(lane_clusters):
        result = fit_lane(lc, degree=args.degree, n_samples=args.n_samples)
        if result is not None:
            fitted_lanes.append(result)
            print(f'        Lane {i+1}: fit OK, {len(result)} sample pts')
        else:
            print(f'        Lane {i+1}: skipped (too short)')

    if not fitted_lanes:
        print('ERROR: No lanes were fitted.')
        return

    # Outputs
    rng_x = tuple(args.range_x)
    rng_y = tuple(args.range_y)

    plot_comparison(raw_pts, fitted_lanes,
                    os.path.join(out_dir, 'lane_fit_comparison.png'),
                    range_x=rng_x, range_y=rng_y)

    plot_fitted_only(fitted_lanes,
                     os.path.join(out_dir, 'lane_fitted_clean.png'),
                     range_x=rng_x, range_y=rng_y)

    save_fitted_json(fitted_lanes, os.path.join(out_dir, 'fitted_lanes.json'))

    print(f'\nDone. Output: {out_dir}')


if __name__ == '__main__':
    main()
