import os
import glob
import numpy as np
import open3d as o3d
from tqdm import tqdm
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from scipy.spatial.transform import Rotation

def _compute_timestamps(points):
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    timestamps = (azimuth - azimuth.min()) / (azimuth.max() - azimuth.min() + 1e-10)
    return timestamps

def get_pose_dict(dataroot):
    config = KISSConfig()
    config.mapping.voxel_size = 0.5
    config.data.max_range = 80.0
    config.data.min_range = 0.5
    icp = KissICP(config=config)
    
    pcd_files = sorted(glob.glob(os.path.join(dataroot, 'os_2_points', '*.pcd')))
    pose_dict = {}
    
    for i, pcf in enumerate(tqdm(pcd_files, desc='KISS-ICP Odometry (U5)')):
        pcd = o3d.io.read_point_cloud(pcf)
        points = np.asarray(pcd.points)
        
        pts64 = points.astype(np.float64)
        ts = _compute_timestamps(pts64)
        icp.register_frame(pts64, ts)
        
        pose = icp.last_pose.copy()
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        q = Rotation.from_matrix(R).as_quat()  # x, y, z, w
        q_wxyz = [q[3], q[0], q[1], q[2]]
        
        frame_id = os.path.splitext(os.path.basename(pcf))[0]
        
        pose_dict[frame_id] = {
            'translation': t.tolist(),
            'rotation': q_wxyz,
            'matrix': pose
        }
        
    return pose_dict
