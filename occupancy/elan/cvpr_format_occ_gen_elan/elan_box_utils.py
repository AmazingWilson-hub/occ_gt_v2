import numpy as np
from scipy.spatial.transform import Rotation

# Class map per ELAN:
# Vehicle -> car (4)
# Pedestrian -> pedestrian (7)
# Cyclist -> bicycle (2)

ELAN_TO_OCC3D = {
    'Vehicle': 4,
    'Pedestrian': 7,
    'Cyclist': 2
}

def fill_box_interior(occ, boxes, labels, gt_bounds, gt_voxel, gt_grid):
    min_bound = np.array(gt_bounds[:3])
    filled_count = 0
    # boxes are [x, y, z, dx, dy, dz, heading] in LiDAR frame?
    for box, label in zip(boxes, labels):
        occ3d_label = ELAN_TO_OCC3D.get(label, 0)
        if occ3d_label == 0: continue
        
        center = box[:3]
        half_ext = box[3:6] / 2
        heading = box[6]
        
        # ELAN might be dx,dy,dz or l,w,h. If x is fwd, typically length=dx
        R_ego = Rotation.from_euler('z', heading).as_matrix()
        
        # 8 corners in box local coords
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
            # Always fill over any background
            occ[idx[0], idx[1], idx[2]] = occ3d_label
            filled_count += 1
            
    return occ, filled_count

def points_in_boxes(points, boxes_lidar):
    # points: (N, 3), boxes_lidar: (K, 7)
    # returns boolean mask (N,) True if in any box
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
