#!/usr/bin/env python3
"""
cvpr_format_occ_gen_egotest — 批次評估所有 Backend

對每個 backend 的輸出計算：
  1. vs 官方 Occ3D GT 的 mIoU（主指標）
  2. vs gt_pose 輸出的 mIoU（看接近上限多少）

使用方式：
  python3 evaluate.py --scene scene-0061
  python3 evaluate.py --scene scene-0061 --backends gt_pose full_fusion kiss_icp
"""

import os
import sys
import argparse
import numpy as np

GT_ROOT    = '/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ/gts'
OUT_ROOT   = os.path.join(os.path.dirname(__file__), 'output')
DATAROOT   = '/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ'

CLASS_NAMES = [
    'noise', 'barrier', 'bicycle', 'bus', 'car', 'const_veh',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'drv_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free',
]

# 官方評估只計算 1-16（非 noise、非 free）
EVAL_CLASSES = list(range(1, 17))


def compute_miou(pred, gt, eval_classes=EVAL_CLASSES):
    """計算指定類別的 mIoU（跳過 union=0 的類別）"""
    ious = {}
    for c in eval_classes:
        pred_c = (pred == c)
        gt_c   = (gt == c)
        inter  = np.sum(pred_c & gt_c)
        union  = np.sum(pred_c | gt_c)
        if union > 0:
            ious[c] = inter / union
    miou = float(np.mean(list(ious.values()))) if ious else 0.0
    return miou, ious


def load_occ(path):
    return np.load(path)['semantics']


def get_frame_tokens(scene_name):
    """從 NuScenes 取得 scene 的所有 keyframe token（有序）"""
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT)
    scene = [s for s in nusc.scene if s['name'] == scene_name][0]
    tokens = []
    tok = scene['first_sample_token']
    while tok:
        sample = nusc.get('sample', tok)
        tokens.append(sample['token'])
        tok = sample['next'] if sample['next'] else None
    return tokens


def evaluate_backend(backend, scene_name, tokens, reference_occs=None):
    """
    評估單一 backend 對官方 GT 的 mIoU。
    reference_occs: 若提供，也計算 vs reference 的 mIoU（通常是 gt_pose）
    """
    pred_dir = os.path.join(OUT_ROOT, backend, scene_name)
    if not os.path.isdir(pred_dir):
        print(f"  [skip] {backend}: 輸出不存在 → {pred_dir}")
        return None

    frame_mious_vs_gt   = []
    frame_mious_vs_ref  = []
    per_class_ious_gt   = {c: [] for c in EVAL_CLASSES}
    pred_occs = []

    missing = 0
    for token in tokens:
        pred_path = os.path.join(pred_dir, token, 'labels.npz')
        gt_path   = os.path.join(GT_ROOT, scene_name, token, 'labels.npz')

        if not os.path.exists(pred_path):
            missing += 1
            continue
        if not os.path.exists(gt_path):
            continue

        pred = load_occ(pred_path)
        gt   = load_occ(gt_path)

        miou, ious = compute_miou(pred, gt)
        frame_mious_vs_gt.append(miou)
        pred_occs.append(pred)
        for c, iou in ious.items():
            per_class_ious_gt[c].append(iou)

        if reference_occs is not None:
            ref_idx = len(pred_occs) - 1
            if ref_idx < len(reference_occs):
                miou_ref, _ = compute_miou(pred, reference_occs[ref_idx])
                frame_mious_vs_ref.append(miou_ref)

    if missing:
        print(f"  [warn] {backend}: {missing} frames missing")

    mean_miou_gt  = float(np.mean(frame_mious_vs_gt)) if frame_mious_vs_gt  else 0.0
    mean_miou_ref = float(np.mean(frame_mious_vs_ref)) if frame_mious_vs_ref else None

    # per-class 平均
    per_class_mean = {}
    for c in EVAL_CLASSES:
        vals = per_class_ious_gt[c]
        per_class_mean[c] = float(np.mean(vals)) if vals else float('nan')

    return {
        'backend':            backend,
        'n_frames':           len(frame_mious_vs_gt),
        'miou_vs_occ3d':      mean_miou_gt,
        'miou_vs_ref':        mean_miou_ref,
        'per_class':          per_class_mean,
        'pred_occs':          pred_occs,
    }


def print_summary(results):
    print(f"\n{'='*70}")
    print(f"  {'Backend':<16s}  {'mIoU(Occ3D GT)':>14s}  {'mIoU(vs gt_pose)':>16s}  {'Frames':>6s}")
    print(f"  {'-'*64}")
    for r in results:
        if r is None:
            continue
        vs_ref = f"{r['miou_vs_ref']:.4f}" if r['miou_vs_ref'] is not None else "   N/A"
        print(f"  {r['backend']:<16s}  {r['miou_vs_occ3d']:>14.4f}  {vs_ref:>16s}  {r['n_frames']:>6d}")
    print(f"{'='*70}\n")


def print_per_class(results):
    print(f"\n{'='*70}")
    print(f"  Per-class IoU vs Occ3D GT")
    print(f"  {'Class':<18s}" + "".join(f"  {r['backend'][:10]:>10s}" for r in results if r))
    print(f"  {'-'*66}")
    for c in EVAL_CLASSES:
        row = f"  {CLASS_NAMES[c]:<18s}"
        for r in results:
            if r is None:
                continue
            v = r['per_class'].get(c, float('nan'))
            row += f"  {v:>10.4f}" if not np.isnan(v) else f"  {'N/A':>10s}"
        print(row)
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='scene-0061')
    parser.add_argument('--backends', nargs='+',
                        default=['gt_pose', 'full_fusion', 'kiss_icp', 'kiss_slam', 'vgicp_fusion', 'glim'])
    args = parser.parse_args()

    print(f"Loading NuScenes frame tokens for {args.scene}...")
    tokens = get_frame_tokens(args.scene)
    print(f"  {len(tokens)} keyframes")

    results = []

    # 先跑 gt_pose 作為 reference
    gt_pose_result = None
    if 'gt_pose' in args.backends:
        print(f"\nEvaluating: gt_pose")
        gt_pose_result = evaluate_backend('gt_pose', args.scene, tokens, reference_occs=None)
        results.append(gt_pose_result)

    ref_occs = gt_pose_result['pred_occs'] if gt_pose_result else None

    # 其餘 backend
    for backend in args.backends:
        if backend == 'gt_pose':
            continue
        print(f"\nEvaluating: {backend}")
        r = evaluate_backend(backend, args.scene, tokens, reference_occs=ref_occs)
        results.append(r)

    print_summary(results)
    print_per_class(results)

    # 儲存 CSV
    os.makedirs(OUT_ROOT, exist_ok=True)
    csv_path = os.path.join(OUT_ROOT, f'comparison_{args.scene}.csv')
    with open(csv_path, 'w') as f:
        f.write('backend,miou_vs_occ3d,miou_vs_gt_pose,n_frames,' +
                ','.join(CLASS_NAMES[c] for c in EVAL_CLASSES) + '\n')
        for r in results:
            if r is None:
                continue
            vs_ref = f"{r['miou_vs_ref']:.4f}" if r['miou_vs_ref'] is not None else ''
            pc_vals = ','.join(
                f"{r['per_class'].get(c, float('nan')):.4f}" for c in EVAL_CLASSES
            )
            f.write(f"{r['backend']},{r['miou_vs_occ3d']:.4f},{vs_ref},{r['n_frames']},{pc_vals}\n")
    print(f"  Comparison CSV saved → {csv_path}")


if __name__ == '__main__':
    main()
