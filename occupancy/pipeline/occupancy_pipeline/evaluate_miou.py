#!/usr/bin/env python3
"""
mIoU Evaluation for GT-compatible Occupancy
Both pred and GT are in the same format: (200, 200, 16) dense grid
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

# 17 classes (0-16) + 17=free/empty
CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]
NUM_CLASSES = 17  # 0-16 are semantic, 17 is free

def compute_iou(pred, gt, mask, num_classes=17):
    """Compute per-class IoU"""
    iou = np.zeros(num_classes)
    
    for c in range(num_classes):
        pred_c = (pred == c) & mask
        gt_c = (gt == c) & mask
        
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        
        if union > 0:
            iou[c] = intersection / union
        else:
            iou[c] = np.nan  # No samples for this class
    
    return iou

def main():
    parser = argparse.ArgumentParser(description="Evaluate mIoU between pred and GT occupancy")
    parser.add_argument('--pred_root', default='/home/t113c52027/t113c52027/occ_gt_v2/occupancy_pipeline/pred_occ/')
    parser.add_argument('--gt_root', default='/data1/nuscenes_occ/gts/')
    parser.add_argument('--scene', default='scene-0061')
    args = parser.parse_args()
    
    pred_scene_dir = os.path.join(args.pred_root, args.scene)
    gt_scene_dir = os.path.join(args.gt_root, args.scene)
    
    if not os.path.exists(pred_scene_dir):
        print(f"Pred dir not found: {pred_scene_dir}")
        return
    if not os.path.exists(gt_scene_dir):
        print(f"GT dir not found: {gt_scene_dir}")
        return
    
    # Find common frames
    pred_tokens = set(os.listdir(pred_scene_dir))
    gt_tokens = set(os.listdir(gt_scene_dir))
    common_tokens = sorted(pred_tokens & gt_tokens)
    
    print(f"Found {len(common_tokens)} common frames to evaluate.")
    
    all_ious = []
    
    for token in tqdm(common_tokens):
        # Load pred
        pred_path = os.path.join(pred_scene_dir, token, 'labels.npz')
        if not os.path.exists(pred_path):
            continue
        pred_data = np.load(pred_path)
        pred = pred_data['semantics']
        
        # Load GT
        gt_path = os.path.join(gt_scene_dir, token, 'labels.npz')
        if not os.path.exists(gt_path):
            continue
        gt_data = np.load(gt_path)
        gt = gt_data['semantics']
        mask = gt_data['mask_lidar'].astype(bool)
        
        # Compute IoU
        iou = compute_iou(pred, gt, mask, NUM_CLASSES)
        all_ious.append(iou)
    
    # Average IoU across all frames
    all_ious = np.array(all_ious)  # (N, 17)
    mean_iou_per_class = np.nanmean(all_ious, axis=0)
    
    print("\n--- Per Class IoU ---")
    for i, name in enumerate(CLASS_NAMES):
        if np.isnan(mean_iou_per_class[i]):
            print(f"{name:20s}: N/A")
        else:
            print(f"{name:20s}: {mean_iou_per_class[i]:.4f}")
    
    # Mean IoU (exclude NaN)
    valid_ious = mean_iou_per_class[~np.isnan(mean_iou_per_class)]
    mean_iou = np.mean(valid_ious)
    print(f"\nMean IoU ({len(valid_ious)} classes): {mean_iou:.4f}")

if __name__ == "__main__":
    main()
