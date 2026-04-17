
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
print("DEBUG: Direct Voxel Comparison")
print("="*60)

# Load
gt_data = np.load(gt_path)
gt_sem = gt_data['semantics']  # (200, 200, 16)
gt_mask = gt_data['mask_lidar']

pred_data = np.load(pred_path)
pred_idx = pred_data['indices']
pred_sem_raw = pred_data['semantics']
pred_sem = LIDARSEG_MAP[np.clip(pred_sem_raw, 0, 31)]

# Convert pred to GT grid
pred_world = np.array(PRED_BOUNDS[:3]) + pred_idx * PRED_VOXEL + 0.5 * PRED_VOXEL
gt_min = np.array(GT_BOUNDS[:3])
gt_max = np.array(GT_BOUNDS[3:])

mask_in_gt = np.all((pred_world >= gt_min) & (pred_world < gt_max), axis=1)
valid_world = pred_world[mask_in_gt]
valid_sem = pred_sem[mask_in_gt]

gt_indices = ((valid_world - gt_min) / GT_VOXEL).astype(int)

# Build pred dense grid
pred_dense = np.ones((200, 200, 16), dtype=np.uint8) * 17
pred_dense[gt_indices[:, 0], gt_indices[:, 1], gt_indices[:, 2]] = valid_sem

# Compare only where GT mask is valid
valid_mask = (gt_mask == 1)

gt_valid = gt_sem[valid_mask]
pred_valid = pred_dense[valid_mask]

print(f"\n[Valid Voxels] Total: {valid_mask.sum()}")
print(f"[GT] Label distribution in valid region:")
for lbl in np.unique(gt_valid):
    cnt = (gt_valid == lbl).sum()
    print(f"  Label {lbl:2d}: {cnt:6d} ({100*cnt/len(gt_valid):.1f}%)")

print(f"\n[Pred] Label distribution in valid region:")
for lbl in np.unique(pred_valid):
    cnt = (pred_valid == lbl).sum()
    print(f"  Label {lbl:2d}: {cnt:6d} ({100*cnt/len(pred_valid):.1f}%)")

# Direct match
match = (gt_valid == pred_valid).sum()
print(f"\n[Match] Exact label match: {match} / {len(gt_valid)} ({100*match/len(gt_valid):.2f}%)")

# Non-empty match (both predict occupied, regardless of class)
gt_occupied = (gt_valid != 17)
pred_occupied = (pred_valid != 17)
both_occupied = gt_occupied & pred_occupied
print(f"\n[Occupancy] GT occupied: {gt_occupied.sum()}")
print(f"[Occupancy] Pred occupied: {pred_occupied.sum()}")
print(f"[Occupancy] Both occupied: {both_occupied.sum()}")

# Check if axes might be swapped - try different orderings
print("\n" + "="*60)
print("Testing different axis orderings...")
print("="*60)

for order_name, order in [("XYZ", (0,1,2)), ("YXZ", (1,0,2)), ("XZY", (0,2,1))]:
    test_idx = gt_indices[:, order]
    # Clip to valid range
    test_idx = np.clip(test_idx, 0, np.array(gt_sem.shape)[list(order)] - 1)
    
    try:
        test_dense = np.ones((200, 200, 16), dtype=np.uint8) * 17
        test_dense[test_idx[:, 0], test_idx[:, 1], test_idx[:, 2]] = valid_sem
        test_pred = test_dense[valid_mask]
        match = (gt_valid == test_pred).sum()
        print(f"  {order_name}: Match = {match} / {len(gt_valid)} ({100*match/len(gt_valid):.2f}%)")
    except Exception as e:
        print(f"  {order_name}: Error - {e}")
