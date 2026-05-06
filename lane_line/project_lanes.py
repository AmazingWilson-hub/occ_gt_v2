#!/usr/bin/env python3
"""
Project fitted lane lines onto camera images and write a video.

Usage:
    python3 lane_line/project_lanes.py \
        --scene highway_sunny_day_2026-04-20-12-58-47 \
        --cam main \
        --out video_out/lane_proj_main.mp4
"""

import os, json, pickle, argparse
import numpy as np
import cv2

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

LANE_COLORS = [
    (  0, 255,   0),   # green
    (  0, 255, 255),   # yellow
    (255,   0,   0),   # blue
    (  0, 128, 255),   # orange
    (255,   0, 255),   # magenta
]


def compose_rpy(roll_deg, pitch_deg, yaw_deg):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) — ZYX convention."""
    r, p, y = np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0         ],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,          1]])
    return Rz @ Ry @ Rx


def world_to_ego(pts_world, T_inv):
    """Nx3 world frame → Nx3 LiDAR/ego frame."""
    ones = np.ones((len(pts_world), 1))
    return (T_inv @ np.hstack([pts_world, ones]).T).T[:, :3]


def project_and_draw_lane(img, pts_lidar, R_cam, t_cam, K, D, W, H, color, thickness):
    """Project and draw a lane line segment-by-segment.

    Only draws a segment when BOTH endpoints are in front of the camera
    and within image bounds — prevents hooks at boundary crossings.
    """
    # fitted_lanes uses y=right-positive; camera calib uses y=left-positive → flip y
    pts_lidar = pts_lidar * np.array([1, -1, 1])
    pts_cam = (R_cam @ pts_lidar.T).T + t_cam   # Nx3 camera frame

    # Project all points at once.
    # Also exclude points beyond the valid distortion range: Brown-Conrady with
    # k1=-0.468 flips sign at r≈2.1 (≈64°), projecting wide-angle points to
    # the wrong side of the image and causing hooks.
    uvs = np.full((len(pts_lidar), 2), -1, dtype=np.float32)
    z = pts_cam[:, 2]
    tan_x = np.abs(pts_cam[:, 0]) / np.maximum(z, 1e-6)
    tan_y = np.abs(pts_cam[:, 1]) / np.maximum(z, 1e-6)
    front = (z > 0.1) & (tan_x < 1.8) & (tan_y < 1.8)
    if front.any():
        imgpts, _ = cv2.projectPoints(
            pts_cam[front].astype(np.float64),
            np.zeros(3), np.zeros(3), K, D
        )
        uvs[front] = imgpts.reshape(-1, 2)

    in_bounds = (front &
                 (uvs[:, 0] >= 0) & (uvs[:, 0] < W) &
                 (uvs[:, 1] >= 0) & (uvs[:, 1] < H))

    for i in range(len(pts_lidar) - 1):
        if in_bounds[i] and in_bounds[i + 1]:
            cv2.line(img,
                     tuple(uvs[i].astype(np.int32)),
                     tuple(uvs[i + 1].astype(np.int32)),
                     color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene',  default='highway_sunny_day_2026-04-20-12-58-47')
    parser.add_argument('--cam',    default='main',
                        choices=['main', 'left', 'right', 'rear', 'sideL', 'sideR'])
    parser.add_argument('--calib',  default=None,
                        help='Path to calibration JSON (default: data/roadlane/calib/config_g6_6view_for0408.json)')
    parser.add_argument('--out',    default=None)
    parser.add_argument('--fps',    type=int, default=10)
    parser.add_argument('--thickness', type=int, default=4)
    args = parser.parse_args()

    scene_dir  = os.path.join(REPO, 'data', 'roadlane', '0429', args.scene)
    lane_json  = os.path.join(REPO, 'lane_line', 'output', 'fitted',
                              args.scene, 'fitted_lanes.json')
    pose_pkl   = os.path.join(REPO, 'occupancy', 'g6', 'cvpr_format_occ_gen_g6',
                              'output', args.scene, 'pose_dict.pkl')
    calib_json = args.calib or os.path.join(REPO, 'data', 'roadlane', 'calib',
                                             'config_g6_6view_for0408.json')
    img_dir    = os.path.join(scene_dir, 'paired', 'images', args.cam)
    out_path   = args.out or os.path.join(REPO, 'video_out',
                                           f'{args.scene}_lane_proj_{args.cam}.mp4')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    with open(lane_json) as f:
        lanes = json.load(f)
    lane_pts_world = [np.array(l['points']) for l in lanes]
    print(f'Lane lines: {len(lane_pts_world)}')

    with open(pose_pkl, 'rb') as f:
        pose_dict = pickle.load(f)

    with open(calib_json) as f:
        calib = json.load(f)

    # ── Camera parameters ──────────────────────────────────────────────────────
    cam_cfg = calib['cameras'][f'port_{args.cam}']
    intr    = cam_cfg['intrinsic']
    extr    = cam_cfg['extrinsic']

    K = np.array([[intr['fx'], 0,          intr['cx']],
                  [0,          intr['fy'], intr['cy']],
                  [0,          0,          1         ]], dtype=np.float64)
    D = np.array([intr['k1'], intr['k2'], intr['p1'],
                  intr['p2'], intr['k3']], dtype=np.float64)

    # x,y,z in config are cm → convert to metres
    t_cam = np.array([extr['x'], extr['y'], extr['z']], dtype=np.float64) / 100.0
    R_cam = compose_rpy(extr['roll'], extr['pitch'], extr['yaw'])

    print(f'Camera: {args.cam}  |  t_cam (m): {t_cam.round(3)}')

    # ── Frame loop ─────────────────────────────────────────────────────────────
    frame_ids  = sorted(pose_dict.keys())
    img_files  = sorted(os.listdir(img_dir))
    n_frames   = min(len(frame_ids), len(img_files))
    print(f'Rendering {n_frames} frames → {out_path}')

    sample = cv2.imread(os.path.join(img_dir, img_files[0]))
    H, W   = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (W, H))

    for i in range(n_frames):
        fid = frame_ids[i]
        img = cv2.imread(os.path.join(img_dir, img_files[i]))
        if img is None:
            continue

        T_inv = np.linalg.inv(pose_dict[fid]['matrix'])  # world → ego

        for li, pts_world in enumerate(lane_pts_world):
            pts_ego = world_to_ego(pts_world, T_inv)
            color   = LANE_COLORS[li % len(LANE_COLORS)]
            project_and_draw_lane(img, pts_ego, R_cam, t_cam, K, D, W, H,
                                  color, args.thickness)

        writer.write(img)

    writer.release()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
