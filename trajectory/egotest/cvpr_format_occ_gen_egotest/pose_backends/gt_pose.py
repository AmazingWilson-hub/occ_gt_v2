"""
Backend: GT Pose (上限基準)
直接使用 NuScenes 官方 ego_pose。
"""

from scipy.spatial.transform import Rotation


def get_pose_dict(nusc, scene_name):
    """
    回傳官方 GT ego_pose。
    Returns: {utime: {'rotation': [w,x,y,z], 'translation': [x,y,z]}}
    """
    scene = [s for s in nusc.scene if s['name'] == scene_name][0]
    pose_dict = {}

    sample_token = scene['first_sample_token']
    while sample_token:
        sample = nusc.get('sample', sample_token)
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego = nusc.get('ego_pose', sd['ego_pose_token'])

        pose_dict[sd['timestamp']] = {
            'rotation': ego['rotation'],       # 已是 [w,x,y,z]
            'translation': ego['translation'],
        }
        sample_token = sample['next'] if sample['next'] else None

    print(f"  [gt_pose] Loaded {len(pose_dict)} GT poses.")
    return pose_dict
