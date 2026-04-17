#!/usr/bin/env python3
"""
cvpr_format_occ_gen_v4 — Dual-Sweep Pipeline
Features:
1. Ego Pose Backend: kiss_slam (default)
2. Sweeps: driveable_surface uses long sweeps (default 40),
           other static classes use short sweeps (default 10)
           to reduce drift accumulation on small objects.
3. Dynamic Object Handling: V2 Box Volume Filling
"""

import os
import sys
import argparse
import importlib
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Add paths to use existing components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'v1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'trajectory', 'egotest', 'cvpr_format_occ_gen_egotest'))

DATAROOT = '/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ'

# --- Occ3D parameters ---
GT_BOUNDS = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
GT_VOXEL  = 0.4
GT_GRID   = (200, 200, 16)

# --- Label Mapping ---
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

# lidarseg label for driveable_surface
DRIVEABLE_SURFACE_LIDARSEG = 24

CATEGORY_TO_OCC3D = {
    'human.pedestrian.adult': 7, 'human.pedestrian.child': 7,
    'human.pedestrian.construction_worker': 7, 'human.pedestrian.police_officer': 7,
    'movable_object.barrier': 1, 'movable_object.trafficcone': 8,
    'vehicle.bicycle': 2, 'vehicle.bus.bendy': 3, 'vehicle.bus.rigid': 3,
    'vehicle.car': 4, 'vehicle.construction': 5, 'vehicle.motorcycle': 6,
    'vehicle.trailer': 9, 'vehicle.truck': 10,
}

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
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_path, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
    lidarseg_file = os.path.join(nusc.dataroot, nusc.get('lidarseg', sample['data']['LIDAR_TOP'])['filename'])
    points_label = np.fromfile(lidarseg_file, dtype=np.uint8)
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd_rec['filename']))
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    gt_pose = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    instance_tokens = [nusc.get('sample_annotation', tok)['instance_token'] for tok in sample['anns']]
    return {
        'pc': pc, 'token': sample['token'],
        'cs_record': cs_record, 'gt_pose': gt_pose, 'utime': sd_rec['timestamp'],
        'lidarseg': points_label, 'boxes': boxes, 'instance_tokens': instance_tokens,
    }

def prev2ego_with_pose(points, prev_cs, prev_pose, ego_cs, ego_pose):
    points = transform(points, quat_wxyz_to_rot(prev_cs['rotation']),  np.array(prev_cs['translation']))
    points = transform(points, quat_wxyz_to_rot(prev_pose['rotation']), np.array(prev_pose['translation']))
    points = transform(points, quat_wxyz_to_rot(ego_pose['rotation']),  np.array(ego_pose['translation']), inverse=True)
    points = transform(points, quat_wxyz_to_rot(ego_cs['rotation']),   np.array(ego_cs['translation']),   inverse=True)
    return points

def keyframe_align_with_pose(prev_info, ego_info, prev_pose, ego_pose):
    from nuscenes.utils.geometry_utils import points_in_box
    pc  = prev_info['pc'].points.copy()
    seg = prev_info['lidarseg'].copy()

    mask_ego = (seg == 31)
    pc  = pc[:, ~mask_ego]
    seg = seg[~mask_ego]

    static_mask = (seg >= 24) & (seg <= 30)
    static_pts  = pc[:, static_mask]
    static_seg  = seg[static_mask]
    static_pts  = prev2ego_with_pose(
        static_pts[:3, :],
        prev_info['cs_record'], prev_pose,
        ego_info['cs_record'],  ego_pose,
    )
    pcs, segs = [static_pts], [static_seg]

    for i, box in enumerate(prev_info['boxes']):
        inst_token = prev_info['instance_tokens'][i]
        if inst_token not in ego_info['instance_tokens']:
            continue
        box_mask = points_in_box(box, prev_info['pc'].points[:3, :])
        if np.sum(box_mask) == 0: continue

        box_p = prev_info['pc'].points[:, box_mask].copy()
        box_s = prev_info['lidarseg'][box_mask].copy()

        cur_idx = ego_info['instance_tokens'].index(inst_token)
        cur_box = ego_info['boxes'][cur_idx]

        box_p = rotate(box_p, np.linalg.inv(box.rotation_matrix), center=box.center)
        box_p = translate(box_p, cur_box.center - box.center)
        box_p = rotate(box_p, cur_box.rotation_matrix, center=cur_box.center)

        pcs.append(box_p[:3, :])
        segs.append(box_s)

    if pcs: return np.concatenate(pcs, axis=-1), np.concatenate(segs)
    return np.zeros((3, 0)), np.zeros(0, dtype=np.uint8)

def fill_box_interior(occ, boxes, cs_record):
    min_bound = np.array(GT_BOUNDS[:3])
    R_l2e = quat_wxyz_to_rot(cs_record['rotation'])
    t_l2e = np.array(cs_record['translation'])
    filled_count = 0

    for box in boxes:
        occ3d_label = CATEGORY_TO_OCC3D.get(box.name, None)
        if occ3d_label is None: continue

        center_ego = R_l2e @ box.center + t_l2e
        rot_ego = R_l2e @ box.rotation_matrix
        half_ext = np.array([box.wlh[1] / 2, box.wlh[0] / 2, box.wlh[2] / 2])

        corners_local = np.array([
            [-half_ext[0], -half_ext[1], -half_ext[2]], [ half_ext[0], -half_ext[1], -half_ext[2]],
            [-half_ext[0],  half_ext[1], -half_ext[2]], [ half_ext[0],  half_ext[1], -half_ext[2]],
            [-half_ext[0], -half_ext[1],  half_ext[2]], [ half_ext[0], -half_ext[1],  half_ext[2]],
            [-half_ext[0],  half_ext[1],  half_ext[2]], [ half_ext[0],  half_ext[1],  half_ext[2]],
        ])
        corners_ego = (rot_ego @ corners_local.T).T + center_ego
        aabb_min, aabb_max = corners_ego.min(axis=0), corners_ego.max(axis=0)

        idx_min = np.clip(np.floor((aabb_min - min_bound) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
        idx_max = np.clip(np.ceil((aabb_max - min_bound) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)

        xs = np.arange(idx_min[0], idx_max[0] + 1)
        ys = np.arange(idx_min[1], idx_max[1] + 1)
        zs = np.arange(idx_min[2], idx_max[2] + 1)
        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0: continue

        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
        voxel_indices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        voxel_centers = voxel_indices * GT_VOXEL + min_bound + GT_VOXEL / 2

        local_coords = (rot_ego.T @ (voxel_centers - center_ego).T).T
        inside = np.all(np.abs(local_coords) <= half_ext, axis=1)

        for idx in voxel_indices[inside]:
            if occ[idx[0], idx[1], idx[2]] == 17:
                occ[idx[0], idx[1], idx[2]] = occ3d_label
                filled_count += 1
    return occ, filled_count

def generate_occupancy(nusc, scene_name, pose_dict, long_sweeps=40, short_sweeps=10):
    """
    long_sweeps:  number of sweeps for driveable_surface (lidarseg=24)
    short_sweeps: number of sweeps for other static classes (lidarseg 25-30)
                  to limit drift accumulation on small objects
    """
    scene = [s for s in nusc.scene if s['name'] == scene_name][0]
    results, tokens = [], []
    sample_token = scene['first_sample_token']

    pbar = tqdm(total=scene['nbr_samples'], desc=f"Generating scene {scene_name}")
    total_filled = 0

    while sample_token:
        sample = nusc.get('sample', sample_token)
        curr_info = get_frame_info(sample, nusc)
        curr_pose = pose_dict[curr_info['utime']] if pose_dict else curr_info['gt_pose']

        pcs, segs = [curr_info['pc'].points[:3, :]], [curr_info['lidarseg']]

        def add_frame(frame_info, frame_pose, frame_idx):
            p, s = keyframe_align_with_pose(frame_info, curr_info, frame_pose, curr_pose)
            # Beyond short_sweeps: keep only driveable_surface
            if frame_idx >= short_sweeps:
                keep = (s == DRIVEABLE_SURFACE_LIDARSEG)
                p, s = p[:, keep], s[keep]
            pcs.append(p)
            segs.append(s)

        # Past frames
        prev_sample = sample
        for i in range(long_sweeps):
            if not prev_sample['prev']: break
            prev_sample = nusc.get('sample', prev_sample['prev'])
            prev_i = get_frame_info(prev_sample, nusc)
            add_frame(prev_i, pose_dict[prev_i['utime']] if pose_dict else prev_i['gt_pose'], i)

        # Future frames
        next_sample = sample
        for i in range(long_sweeps):
            if not next_sample['next']: break
            next_sample = nusc.get('sample', next_sample['next'])
            next_i = get_frame_info(next_sample, nusc)
            add_frame(next_i, pose_dict[next_i['utime']] if pose_dict else next_i['gt_pose'], i)

        # Voxelization
        all_pts, all_lbl = np.concatenate(pcs, axis=-1), np.concatenate(segs)
        cs = curr_info['cs_record']
        xyz_ego = (quat_wxyz_to_rot(cs['rotation']) @ all_pts[:3, :] + np.array(cs['translation']).reshape(3, 1)).T

        min_b, max_b = np.array(GT_BOUNDS[:3]), np.array(GT_BOUNDS[3:])
        mask = np.all((xyz_ego >= min_b) & (xyz_ego < max_b), axis=1)
        xyz, lbls = xyz_ego[mask], LIDARSEG_TO_OCC3D[np.clip(all_lbl[mask], 0, 31)]

        idxs = np.clip(((xyz - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
        occ = np.ones(GT_GRID, dtype=np.uint8) * 17
        occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = lbls

        # V2 addition: fill interior
        occ, filled = fill_box_interior(occ, curr_info['boxes'], curr_info['cs_record'])
        total_filled += filled

        results.append(occ)
        tokens.append(sample['token'])
        sample_token = sample['next'] if sample['next'] else None
        pbar.update(1)

    pbar.close()
    print(f"Total boxes filled: {total_filled}")
    return results, tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='kiss_slam')
    parser.add_argument('--scene', default='scene-0061')
    parser.add_argument('--long_sweeps', type=int, default=40,
                        help='Sweeps for driveable_surface')
    parser.add_argument('--short_sweeps', type=int, default=10,
                        help='Sweeps for other static classes')
    parser.add_argument('--out_root', default=os.path.join(os.path.dirname(__file__), 'output'))
    args = parser.parse_args()

    nusc = load_nuscenes()
    backend_mod = importlib.import_module(f'pose_backends.{args.backend}')
    pose_dict = backend_mod.get_pose_dict(nusc, args.scene)

    occ_list, token_list = generate_occupancy(nusc, args.scene, pose_dict, args.long_sweeps, args.short_sweeps)

    save_dir = os.path.join(args.out_root, args.backend, args.scene)
    os.makedirs(save_dir, exist_ok=True)
    for occ, token in zip(occ_list, token_list):
        os.makedirs(os.path.join(save_dir, token), exist_ok=True)
        np.savez_compressed(os.path.join(save_dir, token, 'labels.npz'), semantics=occ)

    print(f"Saved {len(occ_list)} frames to {save_dir}")

if __name__ == '__main__':
    main()
