"""
Backend: Full Fusion (現有基準)
重用 pose_estimation/estimate_ego_pose.py 的 method_full_fusion()。
已知 scene-0061 平移誤差 ≈ 0.96m，Occupancy mIoU ≈ 0.4882。
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# 加入 pose_estimation 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'trajectory', 'estimation', 'pose_estimation'))


def get_pose_dict(nusc, scene_name):
    """
    用 Full Fusion 推算 ego pose。
    Returns: {utime: {'rotation': [w,x,y,z], 'translation': [x,y,z]}}
    """
    from estimate_ego_pose import (
        load_canbus, get_scene_gt_poses, get_scene_lidar_paths,
        method_full_fusion,
    )

    print("  [full_fusion] Loading sensor data...")
    gt_poses = get_scene_gt_poses(nusc, scene_name)
    canbus_imu = load_canbus(scene_name, 'ms_imu')
    canbus_pose = load_canbus(scene_name, 'pose')
    lidar_infos = get_scene_lidar_paths(nusc, scene_name)
    cs_record = lidar_infos[0]['cs_record']

    print("  [full_fusion] Running Full Fusion estimation...")
    estimated_poses = method_full_fusion(
        lidar_infos, canbus_imu, canbus_pose, gt_poses, cs_record
    )

    pose_dict = {}
    for i, gt in enumerate(gt_poses):
        T = estimated_poses[i]
        R = Rotation.from_matrix(T[:3, :3])
        q = R.as_quat()  # scipy: xyzw
        pose_dict[gt['utime']] = {
            'rotation': [q[3], q[0], q[1], q[2]],  # → wxyz
            'translation': T[:3, 3].tolist(),
        }

    print(f"  [full_fusion] Done. {len(pose_dict)} poses estimated.")
    return pose_dict
