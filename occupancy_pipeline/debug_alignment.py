
import numpy as np
import os

# Paths
pred_path = "/home/t113c52027/t113c52027/occ_gt_v2/presentation_exporter/output_demo/scene-0061/000_ca9a282c9e77460f8360f564131a8af5/OCCUPANCY.npz"
gt_path = "/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ/gts/scene-0061/ca9a282c9e77460f8360f564131a8af5/labels.npz"

# --- CONFIG (from evaluate_miou.py) ---
GT_BOUNDS = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
GT_VOXEL = 0.4
PRED_BOUNDS = [-60.0, -60.0, -5.0, 60.0, 60.0, 11.0]
PRED_VOXEL = 0.2

# Label Mapping
LIDARSEG_MAP = np.zeros(32, dtype=np.uint8)
LIDARSEG_MAP[:] = 0
LIDARSEG_MAP[2:9] = 7  # Pedestrians
LIDARSEG_MAP[9] = 1    # Barrier
LIDARSEG_MAP[12] = 8   # Traffic Cone
LIDARSEG_MAP[14] = 2   # Bicycle
LIDARSEG_MAP[15:17] = 3  # Bus
LIDARSEG_MAP[17] = 4   # Car
LIDARSEG_MAP[18] = 5   # Construction Vehicle
LIDARSEG_MAP[21] = 6   # Motorcycle
LIDARSEG_MAP[22] = 9   # Trailer
LIDARSEG_MAP[23] = 10  # Truck
LIDARSEG_MAP[24] = 11  # Driveable Surface
LIDARSEG_MAP[25] = 12  # Other Flat
LIDARSEG_MAP[26] = 13  # Sidewalk
LIDARSEG_MAP[27] = 14  # Terrain
LIDARSEG_MAP[28] = 15  # Manmade
LIDARSEG_MAP[30] = 16  # Vegetation

print("="*60)
print("DEBUG: mIoU Alignment Issue")
print("="*60)

# --- Load GT ---
gt_data = np.load(gt_path)
gt_sem = gt_data['semantics']
gt_mask = gt_data['mask_lidar']

print(f"\n[GT] Shape: {gt_sem.shape}")
print(f"[GT] Unique Labels: {np.unique(gt_sem)}")
print(f"[GT] Mask Shape: {gt_mask.shape}, Sum (valid voxels): {gt_mask.sum()}")

# GT label distribution (only in valid mask)
gt_valid = gt_sem[gt_mask == 1]
print(f"[GT] Valid Voxel Labels: {np.unique(gt_valid, return_counts=True)}")

# --- Load Pred ---
pred_data = np.load(pred_path)
pred_idx = pred_data['indices']
pred_sem_raw = pred_data['semantics']

print(f"\n[Pred] Indices Shape: {pred_idx.shape}")
print(f"[Pred] Raw Labels (before mapping): {np.unique(pred_sem_raw)}")

# Map Labels
pred_sem = LIDARSEG_MAP[np.clip(pred_sem_raw, 0, 31)]
print(f"[Pred] Mapped Labels (after mapping): {np.unique(pred_sem)}")

# --- Check Spatial Overlap ---
# Convert pred indices to world coordinates
pred_world = np.array(PRED_BOUNDS[:3]) + pred_idx * PRED_VOXEL + 0.5 * PRED_VOXEL

print(f"\n[Pred] World Coords Range:")
print(f"  X: {pred_world[:, 0].min():.2f} ~ {pred_world[:, 0].max():.2f}")
print(f"  Y: {pred_world[:, 1].min():.2f} ~ {pred_world[:, 1].max():.2f}")
print(f"  Z: {pred_world[:, 2].min():.2f} ~ {pred_world[:, 2].max():.2f}")

print(f"\n[GT] Expected World Coords Range (from bounds):")
print(f"  X: {GT_BOUNDS[0]:.2f} ~ {GT_BOUNDS[3]:.2f}")
print(f"  Y: {GT_BOUNDS[1]:.2f} ~ {GT_BOUNDS[4]:.2f}")
print(f"  Z: {GT_BOUNDS[2]:.2f} ~ {GT_BOUNDS[5]:.2f}")

# Filter pred to GT bounds
gt_min = np.array(GT_BOUNDS[:3])
gt_max = np.array(GT_BOUNDS[3:])
mask_in_gt = np.all((pred_world >= gt_min) & (pred_world < gt_max), axis=1)
print(f"\n[Overlap] Pred points inside GT bounds: {mask_in_gt.sum()} / {len(pred_idx)}")

if mask_in_gt.sum() > 0:
    # Convert to GT indices
    valid_world = pred_world[mask_in_gt]
    gt_indices = ((valid_world - gt_min) / GT_VOXEL).astype(int)
    print(f"[Overlap] GT index range after conversion:")
    print(f"  X: {gt_indices[:, 0].min()} ~ {gt_indices[:, 0].max()} (expected 0~199)")
    print(f"  Y: {gt_indices[:, 1].min()} ~ {gt_indices[:, 1].max()} (expected 0~199)")
    print(f"  Z: {gt_indices[:, 2].min()} ~ {gt_indices[:, 2].max()} (expected 0~15)")
    
    # Sample comparison
    print(f"\n[Sample] First 5 converted GT indices:")
    print(gt_indices[:5])
