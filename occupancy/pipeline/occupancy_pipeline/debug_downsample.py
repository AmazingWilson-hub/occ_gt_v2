
import numpy as np

# Paths
pred_path = "/home/t113c52027/t113c52027/occ_gt_v2/presentation_exporter/output_demo/scene-0061/000_ca9a282c9e77460f8360f564131a8af5/OCCUPANCY.npz"
gt_path = "/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ/gts/scene-0061/ca9a282c9e77460f8360f564131a8af5/labels.npz"

# Config
GT_BOUNDS = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
GT_VOXEL = 0.4
PRED_BOUNDS = [-60.0, -60.0, -5.0, 60.0, 60.0, 11.0]
PRED_VOXEL = 0.2

# Label Mapping
LIDARSEG_MAP = np.zeros(32, dtype=np.uint8)
LIDARSEG_MAP[:] = 0
LIDARSEG_MAP[2:9] = 7; LIDARSEG_MAP[9] = 1; LIDARSEG_MAP[12] = 8
LIDARSEG_MAP[14] = 2; LIDARSEG_MAP[15:17] = 3; LIDARSEG_MAP[17] = 4
LIDARSEG_MAP[18] = 5; LIDARSEG_MAP[21] = 6; LIDARSEG_MAP[22] = 9
LIDARSEG_MAP[23] = 10; LIDARSEG_MAP[24] = 11; LIDARSEG_MAP[25] = 12
LIDARSEG_MAP[26] = 13; LIDARSEG_MAP[27] = 14; LIDARSEG_MAP[28] = 15
LIDARSEG_MAP[30] = 16

print("="*60)
print("Step-by-Step Downsampling Analysis")
print("="*60)

# Load pred
pred_data = np.load(pred_path)
pred_idx = pred_data['indices']
pred_sem_raw = pred_data['semantics']
pred_sem = LIDARSEG_MAP[np.clip(pred_sem_raw, 0, 31)]

print(f"\n[Step 1] Original: {len(pred_idx)} voxels (0.2m grid)")
print(f"  Unique MAPPED labels: {np.unique(pred_sem)}")

# Convert to world
pred_world = np.array(PRED_BOUNDS[:3]) + pred_idx * PRED_VOXEL + 0.5 * PRED_VOXEL

# Filter to GT bounds
gt_min = np.array(GT_BOUNDS[:3])
gt_max = np.array(GT_BOUNDS[3:])
mask_in_gt = np.all((pred_world >= gt_min) & (pred_world < gt_max), axis=1)

valid_world = pred_world[mask_in_gt]
valid_sem = pred_sem[mask_in_gt]
print(f"\n[Step 2] In GT bounds: {len(valid_world)} voxels")
print(f"  Label distribution:")
for lbl in np.unique(valid_sem):
    cnt = (valid_sem == lbl).sum()
    print(f"    Label {lbl:2d}: {cnt}")

# Convert to GT indices
gt_indices = ((valid_world - gt_min) / GT_VOXEL).astype(int)
print(f"\n[Step 3] Converted to GT grid indices")
print(f"  Index range: X={gt_indices[:,0].min()}-{gt_indices[:,0].max()}, "
      f"Y={gt_indices[:,1].min()}-{gt_indices[:,1].max()}, "
      f"Z={gt_indices[:,2].min()}-{gt_indices[:,2].max()}")

# How many UNIQUE gt indices?
unique_gt_idx = np.unique(gt_indices, axis=0)
print(f"\n[Step 4] Unique GT grid cells filled: {len(unique_gt_idx)}")
print(f"  (Multiple 0.2m voxels can map to same 0.4m cell)")
print(f"  Compression ratio: {len(valid_world)} -> {len(unique_gt_idx)} = {len(unique_gt_idx)/len(valid_world)*100:.1f}%")

# Load GT and check overlap
gt_data = np.load(gt_path)
gt_sem = gt_data['semantics']
gt_mask = gt_data['mask_lidar']

print(f"\n[Step 5] GT mask analysis")
print(f"  GT valid voxels (mask==1): {gt_mask.sum()}")

# Build pred dense
pred_dense = np.ones((200, 200, 16), dtype=np.uint8) * 17
pred_dense[gt_indices[:, 0], gt_indices[:, 1], gt_indices[:, 2]] = valid_sem

# Count non-empty in pred_dense
pred_occupied = (pred_dense != 17).sum()
print(f"\n[Step 6] Pred dense grid stats")
print(f"  Total cells in pred_dense: {200*200*16}")
print(f"  Occupied cells (!=17): {pred_occupied}")

# Check overlap with GT mask
pred_in_mask = pred_dense[gt_mask == 1]
pred_occupied_in_mask = (pred_in_mask != 17).sum()
print(f"\n[Step 7] Overlap with GT valid region")
print(f"  Pred occupied cells in GT valid region: {pred_occupied_in_mask}")
print(f"  This should match the '1345' from before...")

# THE KEY CHECK: Are our predictions at the RIGHT locations?
# Sample some GT occupied locations and see what we predict
gt_occupied_mask = (gt_sem != 17) & (gt_mask == 1)
gt_occ_indices = np.argwhere(gt_occupied_mask)
print(f"\n[Step 8] Location match analysis")
print(f"  GT has {len(gt_occ_indices)} occupied voxels in valid region")

# For each GT occupied voxel, check if we also predict occupied
match_count = 0
for idx in gt_occ_indices[:1000]:  # Check first 1000
    x, y, z = idx
    if pred_dense[x, y, z] != 17:
        match_count += 1
print(f"  In first 1000 GT occupied voxels, we predict occupied: {match_count}")

# What about the reverse? For our occupied predictions, how many match GT?
our_occupied = np.argwhere(pred_dense != 17)
gt_match = 0
for idx in our_occupied[:1000]:
    x, y, z = idx
    if gt_sem[x, y, z] != 17:
        gt_match += 1
print(f"  In first 1000 of our occupied voxels, GT is also occupied: {gt_match}")
