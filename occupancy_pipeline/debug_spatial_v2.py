
import numpy as np

# Compare the spatial distribution of occupied voxels
pred_path = "/home/t113c52027/t113c52027/occ_gt_v2/occupancy_pipeline/pred_occ/scene-0061/ca9a282c9e77460f8360f564131a8af5/labels.npz"
gt_path = "/data1/nuscenes_occ/gts/scene-0061/ca9a282c9e77460f8360f564131a8af5/labels.npz"

pred = np.load(pred_path)['semantics']
gt_data = np.load(gt_path)
gt = gt_data['semantics']
mask = gt_data['mask_lidar']

print("="*60)
print("Spatial Distribution Comparison")
print("="*60)

# Count occupied voxels per Z layer
print("\n[Occupied voxels per Z layer] (excluding free=17)")
print(f"{'Z':>3} | {'Pred':>8} | {'GT (masked)':>12}")
print("-" * 30)

for z in range(16):
    pred_occ = ((pred[:,:,z] != 17)).sum()
    gt_occ = ((gt[:,:,z] != 17) & (mask[:,:,z] == 1)).sum()
    print(f"{z:3d} | {pred_occ:8d} | {gt_occ:12d}")

# Check spatial location of occupied voxels
pred_occupied = np.argwhere(pred != 17)
gt_occupied = np.argwhere((gt != 17) & (mask == 1))

print(f"\n[Pred occupied] Total: {len(pred_occupied)}")
print(f"  X: {pred_occupied[:,0].min()} - {pred_occupied[:,0].max()}")
print(f"  Y: {pred_occupied[:,1].min()} - {pred_occupied[:,1].max()}")
print(f"  Z: {pred_occupied[:,2].min()} - {pred_occupied[:,2].max()}")

print(f"\n[GT occupied] Total: {len(gt_occupied)}")
print(f"  X: {gt_occupied[:,0].min()} - {gt_occupied[:,0].max()}")
print(f"  Y: {gt_occupied[:,1].min()} - {gt_occupied[:,1].max()}")
print(f"  Z: {gt_occupied[:,2].min()} - {gt_occupied[:,2].max()}")

# Check overlap
pred_set = set(map(tuple, pred_occupied))
gt_set = set(map(tuple, gt_occupied))
overlap = len(pred_set & gt_set)
print(f"\n[Overlap] {overlap} voxels are occupied in BOTH pred and GT")
print(f"  As % of pred: {100*overlap/len(pred_set):.1f}%")
print(f"  As % of GT: {100*overlap/len(gt_set):.1f}%")

# Check label match at overlapping locations
if overlap > 0:
    overlap_coords = list(pred_set & gt_set)[:1000]
    match = 0
    for x, y, z in overlap_coords:
        if pred[x, y, z] == gt[x, y, z]:
            match += 1
    print(f"\n[Label match at overlapping voxels] {match}/{len(overlap_coords)} = {100*match/len(overlap_coords):.1f}%")
