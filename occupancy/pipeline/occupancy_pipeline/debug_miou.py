
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_root', required=True)
    parser.add_argument('--gt_root', required=True)
    parser.add_argument('--scene', default='scene-0061')
    args = parser.parse_args()
    
    # 1. Find a pair
    pred_dir = os.path.join(args.pred_root, args.scene)
    gt_dir = os.path.join(args.gt_root, args.scene)
    
    pred_folders = sorted(os.listdir(pred_dir))
    # format: 000_TOKEN
    
    pair = None
    for pf in pred_folders:
        if '_' not in pf: continue
        token = pf.split('_')[1]
        gt_path = os.path.join(gt_dir, token, 'labels.npz')
        if os.path.exists(gt_path):
            pair = (os.path.join(pred_dir, pf, 'OCCUPANCY.npz'), gt_path)
            break
            
    if not pair:
        print("No matching pair found")
        return
        
    pred_path, gt_path = pair
    print(f"Inspecting Pair: \nPred: {pred_path}\nGT: {gt_path}\n")
    
    # Load Pred
    pdata = np.load(pred_path)
    p_idx = pdata['indices']
    p_sem = pdata['semantics']
    
    print("--- Prediction ---")
    print(f"Indices Shape: {p_idx.shape}")
    print(f"Min/Max Indices: {p_idx.min(0)} - {p_idx.max(0)}")
    print(f"Semantics Unique: {np.unique(p_sem)}")
    
    # Load GT
    gdata = np.load(gt_path)
    g_sem = gdata['semantics']
    print("\n--- Ground Truth ---")
    print(f"Shape: {g_sem.shape}")
    print(f"Semantics Unique: {np.unique(g_sem)}")
    print(f"Ignore Label? (17 in unique): {17 in np.unique(g_sem)}")

if __name__ == "__main__":
    main()
