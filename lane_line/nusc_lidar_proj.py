#!/usr/bin/env python3
"""
Project raw LiDAR points onto NuScenes camera images and write a video.

Usage:
    python3 lane_line/nusc_lidar_proj.py \
        --scene_idx 0 \
        --version v1.0-mini \
        --cam CAM_FRONT \
        --out video_out/nusc_lidar_proj.mp4
"""

import os, argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

DATAROOT = '/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ'
REPO     = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def quat_to_rot(q_wxyz):
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()

def make_T(rot_wxyz, trans_xyz):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(rot_wxyz)
    T[:3,  3] = trans_xyz
    return T

def apply_T(T, pts):
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_idx', type=int, default=0)
    parser.add_argument('--version',   default='v1.0-mini')
    parser.add_argument('--cam',       default='CAM_FRONT')
    parser.add_argument('--out',       default=None)
    parser.add_argument('--fps',       type=int, default=2)
    parser.add_argument('--max_dist',  type=float, default=40.0,
                        help='Max LiDAR range to display (m)')
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud

    nusc = NuScenes(version=args.version, dataroot=DATAROOT, verbose=True)
    scene = nusc.scene[args.scene_idx]
    print(f'Scene: {scene["name"]}')

    out_path = args.out or os.path.join(
        REPO, 'video_out', f'nusc_{scene["name"]}_lidar_{args.cam}.mp4')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Collect samples
    samples = []
    token = scene['first_sample_token']
    while token:
        s = nusc.get('sample', token)
        samples.append(s)
        token = s['next']
    print(f'{len(samples)} samples')

    # Get image size
    sd0 = nusc.get('sample_data', samples[0]['data'][args.cam])
    img0 = cv2.imread(os.path.join(nusc.dataroot, sd0['filename']))
    H, W = img0.shape[:2]
    print(f'Image size: {W}x{H}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (W, H))

    for idx, s in enumerate(samples):
        # ── Load LiDAR ───────────────────────────────────────────────────────
        sd_lidar = nusc.get('sample_data', s['data']['LIDAR_TOP'])
        cs_lidar = nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])
        ep_lidar = nusc.get('ego_pose',           sd_lidar['ego_pose_token'])

        pc = LidarPointCloud.from_file(
            os.path.join(nusc.dataroot, sd_lidar['filename']))
        pts_lidar = pc.points[:3, :].T   # Nx3
        intensity  = pc.points[3, :]     # N

        # ── Load camera ──────────────────────────────────────────────────────
        sd_cam = nusc.get('sample_data', s['data'][args.cam])
        cs_cam = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
        ep_cam = nusc.get('ego_pose',          sd_cam['ego_pose_token'])

        img = cv2.imread(os.path.join(nusc.dataroot, sd_cam['filename']))
        if img is None:
            continue

        K = np.array(cs_cam['camera_intrinsic'], dtype=np.float64)

        # ── Transform: LiDAR → ego → world → ego(cam) → cam ─────────────────
        T_lidar_ego   = make_T(cs_lidar['rotation'], cs_lidar['translation'])
        T_ego_world   = make_T(ep_lidar['rotation'], ep_lidar['translation'])
        T_world_ego_cam = np.linalg.inv(make_T(ep_cam['rotation'], ep_cam['translation']))
        T_ego_cam_cam   = np.linalg.inv(make_T(cs_cam['rotation'], cs_cam['translation']))

        T_lidar_to_cam = T_ego_cam_cam @ T_world_ego_cam @ T_ego_world @ T_lidar_ego

        pts_cam = apply_T(T_lidar_to_cam, pts_lidar)

        # Keep points in front of camera within max_dist
        dist = np.linalg.norm(pts_lidar, axis=1)
        front = (pts_cam[:, 2] > 0.5) & (dist < args.max_dist)
        p = pts_cam[front]
        inty = intensity[front]

        # ── Pinhole projection (NuScenes images are undistorted) ──────────────
        u = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
        v = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # ── Draw: colour by depth ─────────────────────────────────────────────
        z_norm = np.clip(p[in_img, 2] / args.max_dist, 0, 1)
        for i, (ui, vi) in enumerate(zip(u[in_img], v[in_img])):
            c = int(z_norm[i] * 255)
            cv2.circle(img, (int(ui), int(vi)), 2, (255 - c, c, 0), -1)

        fid = f'{idx:03d}'
        cv2.putText(img, f'Frame {fid}', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, f'Frame {fid}', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

        writer.write(img)
        print(f'  [{idx+1}/{len(samples)}] {front.sum()} pts projected')

    writer.release()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
