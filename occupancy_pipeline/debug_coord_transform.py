
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# NuScenes LiDAR to Ego offset
# In NuScenes, LiDAR is mounted on the car roof
# Ego coordinate is at the rear axle center

nusc = NuScenes(version='v1.0-trainval', dataroot='/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes/', verbose=False)

# Get a sample
sample = nusc.get('sample', 'ca9a282c9e77460f8360f564131a8af5')
sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])

print("="*60)
print("LiDAR -> Ego Coordinate Transform")
print("="*60)

print(f"\nLiDAR Translation (LiDAR origin in Ego frame):")
print(f"  X: {cs_record['translation'][0]:.4f} m")
print(f"  Y: {cs_record['translation'][1]:.4f} m")
print(f"  Z: {cs_record['translation'][2]:.4f} m")

print(f"\nLiDAR Rotation (quaternion):")
print(f"  {cs_record['rotation']}")

rot = Quaternion(cs_record['rotation']).rotation_matrix
print(f"\nRotation Matrix:")
print(rot)

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Our occupancy is in LiDAR coordinates.
GT is in Ego coordinates.

To convert LiDAR -> Ego:
  point_ego = R @ point_lidar + T

Where:
  R = rotation matrix above
  T = [{cs_record['translation'][0]:.4f}, {cs_record['translation'][1]:.4f}, {cs_record['translation'][2]:.4f}]

The Z offset of ~{cs_record['translation'][2]:.2f}m is significant!
This means:
  - A point at Z=0 in LiDAR is at Z={cs_record['translation'][2]:.2f}m in Ego
  - GT Z bounds: [-1.0, 5.4] in Ego
  - If LiDAR height is ~{cs_record['translation'][2]:.2f}m, then:
    - Ground (Z~0 in Ego) is at Z~{-cs_record['translation'][2]:.2f}m in LiDAR
""")
