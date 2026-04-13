#!/usr/bin/env python3
"""
cvpr_format_occ_gen_egotest — Ego Pose Backend 測試主程式

使用不同 Ego Pose 來源生成 Occupancy 並儲存，供 evaluate.py 比較。

使用方式：
  python3 generate.py --backend gt_pose     --scene scene-0061
  python3 generate.py --backend full_fusion --scene scene-0061
  python3 generate.py --backend kiss_icp    --scene scene-0061
"""

import os
import sys
import argparse
import importlib
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# 加入共用模組路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cvpr_format_occ_gen'))

DATAROOT = '/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ'

# Occ3D 參數（與官方一致）
GT_BOUNDS = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
GT_VOXEL  = 0.4
GT_GRID   = (200, 200, 16)

# NuScenes lidarseg → Occ3D label mapping
LIDARSEG_TO_OCC3D = np.zeros(32, dtype=np.uint8)
LIDARSEG_TO_OCC3D[9]  = 1   # barrier
LIDARSEG_TO_OCC3D[14] = 2   # bicycle
LIDARSEG_TO_OCC3D[15] = 3   # bus (bendy)
LIDARSEG_TO_OCC3D[16] = 3   # bus (rigid)
LIDARSEG_TO_OCC3D[17] = 4   # car
LIDARSEG_TO_OCC3D[18] = 5   # construction_vehicle
LIDARSEG_TO_OCC3D[21] = 6   # motorcycle
LIDARSEG_TO_OCC3D[2]  = 7   # pedestrian (adult)
LIDARSEG_TO_OCC3D[3]  = 7   # pedestrian (child)
LIDARSEG_TO_OCC3D[4]  = 7   # pedestrian (construction_worker)
LIDARSEG_TO_OCC3D[6]  = 7   # pedestrian (police_officer)
LIDARSEG_TO_OCC3D[12] = 8   # traffic_cone
LIDARSEG_TO_OCC3D[22] = 9   # trailer
LIDARSEG_TO_OCC3D[23] = 10  # truck
LIDARSEG_TO_OCC3D[24] = 11  # driveable_surface
LIDARSEG_TO_OCC3D[25] = 12  # other_flat
LIDARSEG_TO_OCC3D[26] = 13  # sidewalk
LIDARSEG_TO_OCC3D[27] = 14  # terrain
LIDARSEG_TO_OCC3D[28] = 15  # manmade
LIDARSEG_TO_OCC3D[30] = 16  # vegetation


def load_nuscenes():
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version='v1.0-mini', dataroot=DATAROOT)


def quat_wxyz_to_rot(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def transform(points, R, t, inverse=False):
    if inverse:
        return R.T @ (points - t.reshape(3, 1))
    return R @ points + t.reshape(3, 1)


def translate(points, x):
    for i in range(3):
        points[i, :] = points[i, :] + x[i]
    return points


def rotate(points, rot_matrix, center=None):
    if center is not None:
        points[:3, :] = np.dot(rot_matrix, points[:3, :] - center[:, None]) + center[:, None]
    else:
        points[:3, :] = np.dot(rot_matrix, points[:3, :])
    return points


def get_frame_info(sample, nusc):
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import points_in_box

    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_path, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
    lidarseg_file = os.path.join(
        nusc.dataroot,
        nusc.get('lidarseg', sample['data']['LIDAR_TOP'])['filename']
    )
    points_label = np.fromfile(lidarseg_file, dtype=np.uint8)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_rec['filename']))
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    gt_pose   = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    instance_tokens = [
        nusc.get('sample_annotation', tok)['instance_token']
        for tok in sample['anns']
    ]
    return {
        'pc': pc,
        'token': sample['token'],
        'cs_record': cs_record,
        'gt_pose': gt_pose,
        'utime': sd_rec['timestamp'],
        'lidarseg': points_label,
        'boxes': boxes,
        'instance_tokens': instance_tokens,
    }


def prev2ego_with_pose(points, prev_cs, prev_pose, ego_cs, ego_pose):
    """LiDAR → Ego → Global → Ego(current) → LiDAR(current)"""
    points = transform(points, quat_wxyz_to_rot(prev_cs['rotation']),  np.array(prev_cs['translation']))
    points = transform(points, quat_wxyz_to_rot(prev_pose['rotation']), np.array(prev_pose['translation']))
    points = transform(points, quat_wxyz_to_rot(ego_pose['rotation']),  np.array(ego_pose['translation']), inverse=True)
    points = transform(points, quat_wxyz_to_rot(ego_cs['rotation']),   np.array(ego_cs['translation']),   inverse=True)
    return points


def keyframe_align_with_pose(prev_info, ego_info, prev_pose, ego_pose):
    from nuscenes.utils.geometry_utils import points_in_box
    pc  = prev_info['pc'].points.copy()
    seg = prev_info['lidarseg'].copy()

    # 移除自車點
    mask_ego = (seg == 31)
    pc  = pc[:, ~mask_ego]
    seg = seg[~mask_ego]

    # 靜態背景
    static_mask = (seg >= 24) & (seg <= 30)
    static_pts  = pc[:, static_mask]
    static_seg  = seg[static_mask]
    static_pts  = prev2ego_with_pose(
        static_pts[:3, :],
        prev_info['cs_record'], prev_pose,
        ego_info['cs_record'],  ego_pose,
    )

    pcs  = [static_pts]
    segs = [static_seg]

    # 動態物件：用 bounding box 對位
    for i, box in enumerate(prev_info['boxes']):
        inst_token = prev_info['instance_tokens'][i]
        if inst_token not in ego_info['instance_tokens']:
            continue

        box_mask = points_in_box(box, prev_info['pc'].points[:3, :])
        if np.sum(box_mask) == 0:
            continue

        box_p = prev_info['pc'].points[:, box_mask].copy()
        box_s = prev_info['lidarseg'][box_mask].copy()

        cur_idx = ego_info['instance_tokens'].index(inst_token)
        cur_box = ego_info['boxes'][cur_idx]

        box_p = rotate(box_p, np.linalg.inv(box.rotation_matrix), center=box.center)
        box_p = translate(box_p, cur_box.center - box.center)
        box_p = rotate(box_p, cur_box.rotation_matrix, center=cur_box.center)

        pcs.append(box_p[:3, :])
        segs.append(box_s)

    if pcs:
        return np.concatenate(pcs, axis=-1), np.concatenate(segs)
    return np.zeros((3, 0)), np.zeros(0, dtype=np.uint8)


def generate_occupancy(nusc, scene_name, pose_dict, num_sweeps=10):
    """
    針對 scene 的每個 keyframe，堆疊 num_sweeps 前後幀並體素化。
    pose_dict: {utime: {'rotation': [w,x,y,z], 'translation': [x,y,z]}}
               None → 使用 GT ego_pose
    """
    scene = [s for s in nusc.scene if s['name'] == scene_name][0]
    results  = []
    tokens   = []

    sample_token = scene['first_sample_token']
    while sample_token:
        sample    = nusc.get('sample', sample_token)
        curr_info = get_frame_info(sample, nusc)
        curr_pose = pose_dict[curr_info['utime']] if pose_dict else curr_info['gt_pose']

        pcs  = [curr_info['pc'].points[:3, :]]
        segs = [curr_info['lidarseg']]

        # 過去幀
        prev_sample = sample
        for _ in range(num_sweeps):
            if not prev_sample['prev']:
                break
            prev_sample = nusc.get('sample', prev_sample['prev'])
            prev_info   = get_frame_info(prev_sample, nusc)
            prev_pose   = pose_dict[prev_info['utime']] if pose_dict else prev_info['gt_pose']
            p, s = keyframe_align_with_pose(prev_info, curr_info, prev_pose, curr_pose)
            pcs.append(p); segs.append(s)

        # 未來幀
        next_sample = sample
        for _ in range(num_sweeps):
            if not next_sample['next']:
                break
            next_sample = nusc.get('sample', next_sample['next'])
            next_info   = get_frame_info(next_sample, nusc)
            next_pose   = pose_dict[next_info['utime']] if pose_dict else next_info['gt_pose']
            p, s = keyframe_align_with_pose(next_info, curr_info, next_pose, curr_pose)
            pcs.append(p); segs.append(s)

        # 體素化
        all_pts = np.concatenate(pcs, axis=-1)
        all_lbl = np.concatenate(segs)

        cs = curr_info['cs_record']
        R  = quat_wxyz_to_rot(cs['rotation'])
        t  = np.array(cs['translation'])
        xyz_ego = (R @ all_pts[:3, :] + t.reshape(3, 1)).T

        min_b = np.array(GT_BOUNDS[:3])
        max_b = np.array(GT_BOUNDS[3:])
        mask  = np.all((xyz_ego >= min_b) & (xyz_ego < max_b), axis=1)
        xyz   = xyz_ego[mask]
        lbls  = LIDARSEG_TO_OCC3D[np.clip(all_lbl[mask], 0, 31)]

        idxs  = ((xyz - min_b) / GT_VOXEL).astype(int)
        idxs  = np.clip(idxs, 0, np.array(GT_GRID) - 1)

        occ = np.ones(GT_GRID, dtype=np.uint8) * 17
        occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = lbls

        results.append(occ)
        tokens.append(sample['token'])
        sample_token = sample['next'] if sample['next'] else None

    return results, tokens


def main():
    parser = argparse.ArgumentParser(description="Ego Pose Backend 測試 — Occupancy 生成")
    parser.add_argument('--backend', required=True,
                        choices=['gt_pose', 'full_fusion', 'kiss_icp', 'kiss_slam', 'vgicp_fusion', 'glim'],
                        help="使用的 Ego Pose 來源")
    parser.add_argument('--scene',   default='scene-0061')
    parser.add_argument('--out_root', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--sweeps',  type=int, default=10)
    args = parser.parse_args()

    print(f"Loading NuScenes...")
    nusc = load_nuscenes()

    # 動態 import 對應 backend
    backend_mod = importlib.import_module(f'pose_backends.{args.backend}')
    print(f"\n=== Backend: {args.backend} | Scene: {args.scene} ===")
    pose_dict = backend_mod.get_pose_dict(nusc, args.scene)

    print(f"\nGenerating occupancy (sweeps={args.sweeps})...")
    occ_list, token_list = generate_occupancy(nusc, args.scene, pose_dict, num_sweeps=args.sweeps)

    # 儲存
    save_dir = os.path.join(args.out_root, args.backend, args.scene)
    os.makedirs(save_dir, exist_ok=True)
    for occ, token in zip(occ_list, token_list):
        frame_dir = os.path.join(save_dir, token)
        os.makedirs(frame_dir, exist_ok=True)
        np.savez_compressed(os.path.join(frame_dir, 'labels.npz'), semantics=occ)

    print(f"\nSaved {len(occ_list)} frames → {save_dir}/")


if __name__ == '__main__':
    main()
