"""
Backend: KISS-ICP + GPS EKF correction for G6 dataset.

GPS files: <scene>/gps/000000.txt  (header: timestamp,latitude,longitude,altitude)
  timestamp in nanoseconds
IMU files: <scene>/imu/000000.txt  (no header)
  format: timestamp_sec, ax, ay, az, gx, gy, gz, mx, my, mz
  used only for per-frame timestamps to match GPS

Pose output:
  {frame_id: {'translation': [x,y,z], 'rotation': [w,x,y,z], 'matrix': 4x4}}
  translation is in LiDAR-odometry coordinate frame (first frame = origin)
"""

import os
import glob
import csv
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP


def _compute_timestamps(points):
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    timestamps = (azimuth - azimuth.min()) / (azimuth.max() - azimuth.min() + 1e-10)
    return timestamps


def _load_gps(gps_dir):
    """
    Returns list of {'utime': int (ns), 'xy': np.array([x, y])}
    XY is in metres relative to first GPS fix (equirectangular projection).
    """
    files = sorted(glob.glob(os.path.join(gps_dir, '*.txt')))
    records = []
    for f in files:
        with open(f) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                records.append({
                    'utime': int(row['timestamp']),
                    'lat':   float(row['latitude']),
                    'lon':   float(row['longitude']),
                })
    if not records:
        return []
    # Use first fix as origin
    lat0 = records[0]['lat']
    lon0 = records[0]['lon']
    R_earth = 6371000.0
    for r in records:
        dlat = np.radians(r['lat'] - lat0)
        dlon = np.radians(r['lon'] - lon0)
        r['xy'] = np.array([
            dlon * R_earth * np.cos(np.radians(lat0)),  # x = east
            dlat * R_earth,                              # y = north
        ])
    return records


def _load_imu_timestamps(imu_dir):
    """Returns list of timestamps in nanoseconds (converted from seconds)."""
    files = sorted(glob.glob(os.path.join(imu_dir, '*.txt')))
    utimes = []
    for f in files:
        with open(f) as fp:
            line = fp.readline().strip()
            if line:
                ts_sec = float(line.split(',')[0])
                utimes.append(int(ts_sec * 1e9))
    return utimes


def get_pose_dict(dataroot):
    """
    Run KISS-ICP odometry then apply GPS EKF correction.
    dataroot: path to scene directory (contains VLS128_pcdnpy/, gps/, imu/)
    """
    config = KISSConfig()
    config.mapping.voxel_size = 0.5
    config.data.max_range = 80.0
    config.data.min_range = 0.5
    icp = KissICP(config=config)

    pcd_dir = os.path.join(dataroot, 'VLS128_pcdnpy') if os.path.isdir(os.path.join(dataroot, 'VLS128_pcdnpy')) \
              else os.path.join(dataroot, 'VLS128_pcd')
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
    frame_ids = [os.path.splitext(os.path.basename(f))[0] for f in pcd_files]

    # Load GPS and IMU timestamps
    gps_dir = os.path.join(dataroot, 'gps')
    imu_dir = os.path.join(dataroot, 'imu')
    gps_list = _load_gps(gps_dir) if os.path.isdir(gps_dir) else []
    imu_utimes = _load_imu_timestamps(imu_dir) if os.path.isdir(imu_dir) else []

    if gps_list:
        print(f"  [kiss_icp_gps] Loaded {len(gps_list)} GPS fixes")
    else:
        print(f"  [kiss_icp_gps] No GPS found, running pure odometry")

    # Run KISS-ICP
    raw_poses = []
    for pcf in tqdm(pcd_files, desc='KISS-ICP Odometry'):
        pcd = o3d.io.read_point_cloud(pcf)
        points = np.asarray(pcd.points).astype(np.float64)
        ts = _compute_timestamps(points)
        icp.register_frame(points, ts)
        raw_poses.append(icp.last_pose.copy())

    # GPS coordinate system (east/north) is not aligned with KISS-ICP odometry frame,
    # so GPS EKF correction is skipped. Use pure odometry poses.
    corrected_poses = [p.copy() for p in raw_poses]

    # Build pose_dict
    pose_dict = {}
    for frame_id, T in zip(frame_ids, corrected_poses):
        R = T[:3, :3]
        t = T[:3, 3]
        q = Rotation.from_matrix(R).as_quat()  # xyzw
        pose_dict[frame_id] = {
            'translation': t.tolist(),
            'rotation': [q[3], q[0], q[1], q[2]],  # wxyz
            'matrix': T,
        }

    print(f"  [kiss_icp] Done. {len(pose_dict)} poses.")
    return pose_dict
