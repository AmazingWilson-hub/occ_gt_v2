
import numpy as np

gt_path = "/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ/gts/scene-0061/ca9a282c9e77460f8360f564131a8af5/labels.npz"
pred_path = "/home/t113c52027/t113c52027/occ_gt_v2/presentation_exporter/output_demo/scene-0061/000_ca9a282c9e77460f8360f564131a8af5/OCCUPANCY.npz"

GT_BOUNDS = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
GT_VOXEL = 0.4
PRED_BOUNDS = [-60.0, -60.0, -5.0, 60.0, 60.0, 11.0]
PRED_VOXEL = 0.2

# Label Mapping (lidarseg 0-31 -> occ3d 0-16)
LIDARSEG_MAP = np.zeros(32, dtype=np.uint8)
LIDARSEG_MAP[:] = 0
LIDARSEG_MAP[2:9] = 7; LIDARSEG_MAP[9] = 1; LIDARSEG_MAP[12] = 8
LIDARSEG_MAP[14] = 2; LIDARSEG_MAP[15:17] = 3; LIDARSEG_MAP[17] = 4
LIDARSEG_MAP[18] = 5; LIDARSEG_MAP[21] = 6; LIDARSEG_MAP[22] = 9
LIDARSEG_MAP[23] = 10; LIDARSEG_MAP[24] = 11; LIDARSEG_MAP[25] = 12
LIDARSEG_MAP[26] = 13; LIDARSEG_MAP[27] = 14; LIDARSEG_MAP[28] = 15
LIDARSEG_MAP[30] = 16

print("="*60)
print("Spatial Match Analysis")
print("="*60)

# Load GT
gt_data = np.load(gt_path)
gt_sem = gt_data['semantics']
gt_mask = gt_data['mask_lidar']

# Load Pred
pred_data = np.load(pred_path)
pred_idx = pred_data['indices']
pred_sem_raw = pred_data['semantics']
pred_sem = LIDARSEG_MAP[np.clip(pred_sem_raw, 0, 31)]

# Convert pred to world coords
pred_world = np.array(PRED_BOUNDS[:3]) + pred_idx * PRED_VOXEL + 0.5 * PRED_VOXEL

# Filter to GT bounds
gt_min = np.array(GT_BOUNDS[:3])
gt_max = np.array(GT_BOUNDS[3:])
mask_in_gt = np.all((pred_world >= gt_min) & (pred_world < gt_max), axis=1)

valid_world = pred_world[mask_in_gt]
valid_sem = pred_sem[mask_in_gt]

# Convert to GT indices
gt_indices = ((valid_world - gt_min) / GT_VOXEL).astype(int)

print(f"\n[Pred in GT bounds] {len(gt_indices)} voxels")

# Get unique GT indices (due to resolution difference, multiple pred -> one gt)
unique_gt_idx, counts = np.unique(gt_indices, axis=0, return_counts=True)
print(f"[Unique GT cells] {len(unique_gt_idx)} (after merging 0.2m -> 0.4m)")

# Check how many of these fall into GT mask==1
gt_mask_values = gt_mask[unique_gt_idx[:, 0], unique_gt_idx[:, 1], unique_gt_idx[:, 2]]
in_valid = (gt_mask_values == 1).sum()
in_invalid = (gt_mask_values == 0).sum()

print(f"\n[Mask Check]")
print(f"  In GT valid region (mask=1): {in_valid}")
print(f"  In GT invalid region (mask=0): {in_invalid}")
print(f"  Total: {len(unique_gt_idx)}")

# Where is GT mask=1?
gt_valid_indices = np.argwhere(gt_mask == 1)
print(f"\n[GT Valid Region]")
print(f"  Total valid GT voxels: {len(gt_valid_indices)}")
print(f"  X range: {gt_valid_indices[:,0].min()} - {gt_valid_indices[:,0].max()}")
print(f"  Y range: {gt_valid_indices[:,1].min()} - {gt_valid_indices[:,1].max()}")
print(f"  Z range: {gt_valid_indices[:,2].min()} - {gt_valid_indices[:,2].max()}")

# Where are our predictions?
print(f"\n[Our Predictions]")
print(f"  X range: {unique_gt_idx[:,0].min()} - {unique_gt_idx[:,0].max()}")
print(f"  Y range: {unique_gt_idx[:,1].min()} - {unique_gt_idx[:,1].max()}")
print(f"  Z range: {unique_gt_idx[:,2].min()} - {unique_gt_idx[:,2].max()}")

# What Z values are most common in our predictions?
z_counts = np.bincount(unique_gt_idx[:, 2], minlength=16)
print(f"\n[Z Distribution of our predictions]")
for z in range(16):
    print(f"  Z={z}: {z_counts[z]} voxels")

# What Z values are most common in GT valid region?
gt_z_counts = np.bincount(gt_valid_indices[:, 2], minlength=16)
print(f"\n[Z Distribution of GT valid region]")
for z in range(16):
    print(f"  Z={z}: {gt_z_counts[z]} voxels")
