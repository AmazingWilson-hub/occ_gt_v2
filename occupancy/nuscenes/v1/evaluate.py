#!/usr/bin/env python3
"""
mIoU Evaluation: Compare generated occupancy vs CVPR2023 Occ3D GT
Both are dense (200, 200, 16) grids with Occ3D labels (0-16, 17=free)
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]
NUM_CLASSES = 17

def compute_iou(pred, gt, mask, num_classes=17):
    iou = np.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (pred == c) & mask
        gt_c = (gt == c) & mask
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        iou[c] = intersection / union if union > 0 else np.nan
    return iou

def main():
    parser = argparse.ArgumentParser(description="Evaluate mIoU")
    parser.add_argument('--pred_root', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--gt_root', default='/data1/nuscenes_occ/gts/')
    parser.add_argument('--scene', default='scene-0061')
    args = parser.parse_args()
    
    pred_dir = os.path.join(args.pred_root, args.scene)
    gt_dir = os.path.join(args.gt_root, args.scene)
    
    if not os.path.exists(pred_dir):
        print(f"Pred dir not found: {pred_dir}")
        return
    if not os.path.exists(gt_dir):
        print(f"GT dir not found: {gt_dir}")
        return
    
    common = sorted(set(os.listdir(pred_dir)) & set(os.listdir(gt_dir)))
    print(f"Found {len(common)} common frames.")
    
    all_ious = []
    for token in tqdm(common):
        pred_path = os.path.join(pred_dir, token, 'labels.npz')
        gt_path = os.path.join(gt_dir, token, 'labels.npz')
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue
        
        pred = np.load(pred_path)['semantics']
        gt_data = np.load(gt_path)
        gt = gt_data['semantics']
        mask = gt_data['mask_lidar'].astype(bool)
        
        iou = compute_iou(pred, gt, mask, NUM_CLASSES)
        all_ious.append(iou)
    
    all_ious = np.array(all_ious)
    mean_per_class = np.nanmean(all_ious, axis=0)
    
    print("\n--- Per Class IoU ---")
    for i, name in enumerate(CLASS_NAMES):
        val = mean_per_class[i]
        print(f"{name:20s}: {'N/A' if np.isnan(val) else f'{val:.4f}'}")
    
    valid = mean_per_class[~np.isnan(mean_per_class)]
    print(f"\nMean IoU ({len(valid)} classes): {np.mean(valid):.4f}")

if __name__ == "__main__":
    main()
