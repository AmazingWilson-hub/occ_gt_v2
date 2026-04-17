#!/usr/bin/env python3
"""
eval_gt_pose_s10.py — 生成 gt_pose (sweeps=10, no box fill) 並計算 mIoU vs Occ3D GT
"""

import os
import sys
import importlib.util
import numpy as np
from tqdm import tqdm

# Import generate.py from this directory directly to avoid path conflicts
_spec = importlib.util.spec_from_file_location(
    'gen_v3', os.path.join(os.path.dirname(__file__), 'generate.py'))
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)

DATAROOT   = '/data2/t113c52027/occ_gt_v2/data/nuscenes_occ'
GT_ROOT    = '/data2/t113c52027/occ_gt_v2/data/nuscenes_occ/gts'
OUT_DIR    = os.path.join(os.path.dirname(__file__), 'output_s10_nobox', 'gt_pose', 'scene-0061')
SCENE      = 'scene-0061'
NUM_SWEEPS = 10

CLASS_NAMES = [
    'noise', 'barrier', 'bicycle', 'bus', 'car', 'const_veh',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'drv_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free',
]
EVAL_CLASSES = list(range(1, 17))


def compute_miou(pred, gt):
    ious = {}
    for c in EVAL_CLASSES:
        inter = np.sum((pred == c) & (gt == c))
        union = np.sum((pred == c) | (gt == c))
        if union > 0:
            ious[c] = inter / union
    return float(np.mean(list(ious.values()))) if ious else 0.0, ious


def main():
    print(f"Loading NuScenes ({SCENE})...")
    nusc = _gen.load_nuscenes()

    # gt_pose: pose_dict=None → uses gt_pose from nusc directly
    occ_list, token_list = _gen.generate_occupancy(
        nusc, SCENE, pose_dict=None, num_sweeps=NUM_SWEEPS, box_fill=False)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nEvaluating mIoU vs Occ3D GT...")
    ious_all = {c: [] for c in EVAL_CLASSES}
    frame_mious = []

    for occ, token in tqdm(zip(occ_list, token_list), total=len(token_list), desc='Save + Eval'):
        # Save
        frame_dir = os.path.join(OUT_DIR, token)
        os.makedirs(frame_dir, exist_ok=True)
        np.savez_compressed(os.path.join(frame_dir, 'labels.npz'), semantics=occ)

        # Eval
        gt_path = os.path.join(GT_ROOT, SCENE, token, 'labels.npz')
        if os.path.exists(gt_path):
            gt = np.load(gt_path)['semantics']
            miou, ious = compute_miou(occ, gt)
            frame_mious.append(miou)
            for c, v in ious.items():
                ious_all[c].append(v)

    print(f"\nSaved to: {OUT_DIR}")
    print(f"\n{'Class':<18s}  {'IoU':>6s}")
    per_class = {}
    for c in EVAL_CLASSES:
        v = float(np.mean(ious_all[c])) if ious_all[c] else float('nan')
        per_class[c] = v
        print(f"  {CLASS_NAMES[c]:<16s}  {v:.4f}")
    miou = float(np.nanmean(list(per_class.values())))
    print(f"\n  mIoU = {miou:.4f}  ({len(frame_mious)} frames)")


if __name__ == '__main__':
    main()
