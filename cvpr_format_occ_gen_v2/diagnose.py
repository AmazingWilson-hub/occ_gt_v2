#!/usr/bin/env python3
"""Diagnose why mIoU is low despite visual similarity"""

import numpy as np
import os

scene = 'scene-0061'
pred_root = '/home/t113c52027/t113c52027/occ_gt_v2/cvpr_format_occ_gen/output'
gt_root = '/data1/nuscenes_occ/gts'

CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]

# Pick first frame
tokens = sorted(set(os.listdir(f"{pred_root}/{scene}")) & set(os.listdir(f"{gt_root}/{scene}")))
token = tokens[5]  # Use 6th frame

pred = np.load(f"{pred_root}/{scene}/{token}/labels.npz")['semantics']
gt_data = np.load(f"{gt_root}/{scene}/{token}/labels.npz")
gt = gt_data['semantics']
mask = gt_data['mask_lidar'].astype(bool)

print(f"Token: {token}")
print(f"Grid shape: {pred.shape}")
print(f"Mask coverage: {mask.sum()} / {mask.size} ({100*mask.sum()/mask.size:.1f}%)")
print()

# Analysis 1: Overall match
pred_m = pred[mask]
gt_m = gt[mask]
print(f"=== Overall (within mask) ===")
print(f"Exact match: {(pred_m == gt_m).sum()} / {len(pred_m)} ({100*(pred_m == gt_m).sum()/len(pred_m):.1f}%)")
print(f"Both free (17): {((pred_m == 17) & (gt_m == 17)).sum()}")
print(f"Both occupied (!=17): {((pred_m != 17) & (gt_m != 17)).sum()}")
print(f"GT occupied, Pred free: {((pred_m == 17) & (gt_m != 17)).sum()}")
print(f"GT free, Pred occupied: {((pred_m != 17) & (gt_m == 17)).sum()}")
print()

# Analysis 2: Per-class breakdown
print(f"=== Per-class analysis (within mask) ===")
print(f"{'Class':>20} | {'GT count':>8} | {'Pred count':>10} | {'Match':>6} | {'GT occ, Pred free':>18} | {'GT free, Pred occ':>18}")
print("-" * 95)

for c in range(17):
    gt_c = (gt_m == c)
    pred_c = (pred_m == c)
    match = (gt_c & pred_c).sum()
    gt_occ_pred_free = (gt_c & (pred_m == 17)).sum()  # GT says class c, we say free
    gt_free_pred_occ = ((gt_m == 17) & pred_c).sum()  # GT says free, we say class c
    
    print(f"{CLASS_NAMES[c]:>20} | {gt_c.sum():8d} | {pred_c.sum():10d} | {match:6d} | {gt_occ_pred_free:18d} | {gt_free_pred_occ:18d}")

# Analysis 3: For "car" class specifically
print(f"\n=== Detailed 'car' analysis ===")
c = 4  # car
gt_car = (gt == c) & mask
pred_car = (pred == c) & mask
intersection = (gt_car & pred_car).sum()
union = (gt_car | pred_car).sum()
print(f"GT car voxels: {gt_car.sum()}")
print(f"Pred car voxels: {pred_car.sum()}")
print(f"Intersection: {intersection}")
print(f"Union: {union}")
print(f"IoU: {intersection/union:.4f}" if union > 0 else "IoU: N/A")

# Where GT has car, what does pred say?
gt_car_locs = gt_m == c
pred_at_gt_car = pred_m[gt_car_locs]
print(f"\nWhere GT=car, pred says:")
for lbl in sorted(np.unique(pred_at_gt_car)):
    name = CLASS_NAMES[lbl] if lbl < 17 else 'free'
    cnt = (pred_at_gt_car == lbl).sum()
    print(f"  {name:>20}: {cnt} ({100*cnt/len(pred_at_gt_car):.1f}%)")
