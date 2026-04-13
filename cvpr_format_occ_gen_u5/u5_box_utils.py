import numpy as np
from scipy.spatial.transform import Rotation

# U5 3D Box class mapping → Occ3D labels
# 目前 U5 只有 Vehicle 類別
U5_TO_OCC3D = {
    'Vehicle': 4,       # car
    'Pedestrian': 7,    # pedestrian
    'Cyclist': 2,       # bicycle
}

def fill_box_interior(occ, boxes, labels, gt_bounds, gt_voxel, gt_grid):
    min_bound = np.array(gt_bounds[:3])
    filled_count = 0
    for box, label in zip(boxes, labels):
        occ3d_label = U5_TO_OCC3D.get(label, 0)
        if occ3d_label == 0: continue
        
        center = box[:3]
        half_ext = box[3:6] / 2
        heading = box[6]
        
        R_ego = Rotation.from_euler('z', heading).as_matrix()
        
        corners_local = np.array([
            [-half_ext[0], -half_ext[1], -half_ext[2]],
            [ half_ext[0], -half_ext[1], -half_ext[2]],
            [-half_ext[0],  half_ext[1], -half_ext[2]],
            [ half_ext[0],  half_ext[1], -half_ext[2]],
            [-half_ext[0], -half_ext[1],  half_ext[2]],
            [ half_ext[0], -half_ext[1],  half_ext[2]],
            [-half_ext[0],  half_ext[1],  half_ext[2]],
            [ half_ext[0],  half_ext[1],  half_ext[2]],
        ])
        
        corners_ego = (R_ego @ corners_local.T).T + center
        aabb_min = corners_ego.min(axis=0)
        aabb_max = corners_ego.max(axis=0)
        
        idx_min = np.clip(np.floor((aabb_min - min_bound) / gt_voxel).astype(int), 0, np.array(gt_grid) - 1)
        idx_max = np.clip(np.ceil((aabb_max - min_bound) / gt_voxel).astype(int), 0, np.array(gt_grid) - 1)
        
        xs = np.arange(idx_min[0], idx_max[0] + 1)
        ys = np.arange(idx_min[1], idx_max[1] + 1)
        zs = np.arange(idx_min[2], idx_max[2] + 1)
        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0: continue
        
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
        voxel_indices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        voxel_centers = voxel_indices * gt_voxel + min_bound + gt_voxel / 2
        
        local_coords = (R_ego.T @ (voxel_centers - center).T).T
        inside = np.all(np.abs(local_coords) <= half_ext, axis=1)
        
        for idx in voxel_indices[inside]:
            occ[idx[0], idx[1], idx[2]] = occ3d_label
            filled_count += 1
            
    return occ, filled_count

def points_in_boxes(points, boxes_lidar):
    if len(boxes_lidar) == 0:
        return np.zeros(len(points), dtype=bool)
    mask = np.zeros(len(points), dtype=bool)
    for box in boxes_lidar:
        center = box[:3]
        half_ext = box[3:6] / 2
        heading = box[6]
        R = Rotation.from_euler('z', heading).as_matrix()
        local_pts = (R.T @ (points[:, :3] - center).T).T
        in_box = np.all(np.abs(local_pts) <= half_ext, axis=1)
        mask |= in_box
    return mask
