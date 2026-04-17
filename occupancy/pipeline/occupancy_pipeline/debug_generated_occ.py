
import numpy as np
import os

# Check our generated occupancy directly
pred_path = "/home/t113c52027/t113c52027/occ_gt_v2/presentation_exporter/output_demo/scene-0061/000_ca9a282c9e77460f8360f564131a8af5/OCCUPANCY.npz"

pred_data = np.load(pred_path)
pred_idx = pred_data['indices']
pred_sem = pred_data['semantics']

print("="*60)
print("Our Generated Occupancy Stats")
print("="*60)

print(f"\nTotal occupied voxels: {len(pred_idx)}")
print(f"Index shape: {pred_idx.shape}")
print(f"Index range:")
print(f"  X: {pred_idx[:, 0].min()} ~ {pred_idx[:, 0].max()} (grid size: 600)")
print(f"  Y: {pred_idx[:, 1].min()} ~ {pred_idx[:, 1].max()} (grid size: 600)")
print(f"  Z: {pred_idx[:, 2].min()} ~ {pred_idx[:, 2].max()} (grid size: 80)")

print(f"\nLabel distribution:")
for lbl in np.unique(pred_sem):
    cnt = (pred_sem == lbl).sum()
    print(f"  Label {lbl:2d}: {cnt:6d}")

# Now check what's in GT range (convert to world, then filter)
PRED_BOUNDS = [-60.0, -60.0, -5.0, 60.0, 60.0, 11.0]
PRED_VOXEL = 0.2
GT_BOUNDS = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

pred_world = np.array(PRED_BOUNDS[:3]) + pred_idx * PRED_VOXEL + 0.5 * PRED_VOXEL
gt_min = np.array(GT_BOUNDS[:3])
gt_max = np.array(GT_BOUNDS[3:])
mask_in_gt = np.all((pred_world >= gt_min) & (pred_world < gt_max), axis=1)

print(f"\nVoxels in GT bounds: {mask_in_gt.sum()} / {len(pred_idx)}")
print(f"  (GT X: {GT_BOUNDS[0]} ~ {GT_BOUNDS[3]})")
print(f"  (GT Y: {GT_BOUNDS[1]} ~ {GT_BOUNDS[4]})")
print(f"  (GT Z: {GT_BOUNDS[2]} ~ {GT_BOUNDS[5]})")

# Sample: print some world coords
print(f"\nSample world coords (first 10 in GT bounds):")
in_gt_world = pred_world[mask_in_gt][:10]
print(in_gt_world)
