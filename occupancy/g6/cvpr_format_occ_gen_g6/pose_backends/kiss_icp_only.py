"""
Backend: Pure KISS-ICP odometry (no GPS correction).
"""

import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP


def _compute_timestamps(points):
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    return (azimuth - azimuth.min()) / (azimuth.max() - azimuth.min() + 1e-10)


def get_pose_dict(dataroot):
    config = KISSConfig()
    config.mapping.voxel_size = 0.5
    config.data.max_range = 80.0
    config.data.min_range = 0.5
    icp = KissICP(config=config)

    pcd_dir = os.path.join(dataroot, 'VLS128_pcdnpy') if os.path.isdir(os.path.join(dataroot, 'VLS128_pcdnpy')) \
              else os.path.join(dataroot, 'VLS128_pcd')
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
    frame_ids = [os.path.splitext(os.path.basename(f))[0] for f in pcd_files]

    import open3d as o3d
    pose_dict = {}
    for pcf, fid in zip(tqdm(pcd_files, desc='KISS-ICP'), frame_ids):
        pcd = o3d.io.read_point_cloud(pcf)
        points = np.asarray(pcd.points).astype(np.float64)
        ts = _compute_timestamps(points)
        icp.register_frame(points, ts)
        T = icp.last_pose.copy()
        R = T[:3, :3]
        t = T[:3, 3]
        q = Rotation.from_matrix(R).as_quat()
        pose_dict[fid] = {
            'translation': t.tolist(),
            'rotation': [q[3], q[0], q[1], q[2]],
            'matrix': T,
        }

    print(f'  [kiss_icp_only] Done. {len(pose_dict)} poses.')
    return pose_dict
