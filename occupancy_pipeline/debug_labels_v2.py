
import numpy as np

# Compare labels directly
pred_path = "/home/t113c52027/t113c52027/occ_gt_v2/occupancy_pipeline/pred_occ/scene-0061/ca9a282c9e77460f8360f564131a8af5/labels.npz"
gt_path = "/data1/nuscenes_occ/gts/scene-0061/ca9a282c9e77460f8360f564131a8af5/labels.npz"

pred = np.load(pred_path)['semantics']
gt_data = np.load(gt_path)
gt = gt_data['semantics']
mask = gt_data['mask_lidar']

print("="*60)
print("Label Distribution Comparison")
print("="*60)

CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free'
]

print(f"\n{'Label':>3} | {'Name':>20} | {'Pred':>8} | {'GT (masked)':>12}")
print("-" * 55)

for lbl in range(18):
    pred_cnt = (pred == lbl).sum()
    gt_cnt = ((gt == lbl) & (mask == 1)).sum()
    name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"unknown_{lbl}"
    print(f"{lbl:3d} | {name:>20} | {pred_cnt:8d} | {gt_cnt:12d}")

# Check label overlap at same positions
valid_mask = (mask == 1)
pred_valid = pred[valid_mask]
gt_valid = gt[valid_mask]

print(f"\n[Valid region analysis]")
print(f"  Total valid voxels: {valid_mask.sum()}")
print(f"  Exact label match: {(pred_valid == gt_valid).sum()} ({100*(pred_valid == gt_valid).sum()/len(pred_valid):.1f}%)")

# Confusion: where GT is X, what does Pred say?
print(f"\n[Confusion Analysis] (only non-free)")
gt_occupied = (gt_valid != 17)
pred_at_gt_occ = pred_valid[gt_occupied]
gt_at_gt_occ = gt_valid[gt_occupied]

print(f"  GT occupied voxels: {gt_occupied.sum()}")
print(f"  Pred occupied at those locations: {(pred_at_gt_occ != 17).sum()}")
print(f"  Pred free at those locations: {(pred_at_gt_occ == 17).sum()}")
