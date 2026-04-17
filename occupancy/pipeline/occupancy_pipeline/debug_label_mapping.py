
import numpy as np

# Check actual label distribution and mapping
pred_path = "/home/t113c52027/t113c52027/occ_gt_v2/occupancy_pipeline/pred_occ/scene-0061/ca9a282c9e77460f8360f564131a8af5/labels.npz"
gt_path = "/data1/nuscenes_occ/gts/scene-0061/ca9a282c9e77460f8360f564131a8af5/labels.npz"

pred = np.load(pred_path)['semantics']
gt_data = np.load(gt_path)
gt = gt_data['semantics']
mask = gt_data['mask_lidar']

# Our current mapping
LIDARSEG_TO_OCC3D = np.zeros(32, dtype=np.uint8)
LIDARSEG_TO_OCC3D[:] = 0       # Default: others
LIDARSEG_TO_OCC3D[2:9] = 7     # Pedestrian variants -> pedestrian
LIDARSEG_TO_OCC3D[9] = 1       # barrier
LIDARSEG_TO_OCC3D[12] = 8      # traffic_cone
LIDARSEG_TO_OCC3D[14] = 2      # bicycle
LIDARSEG_TO_OCC3D[15:17] = 3   # bus
LIDARSEG_TO_OCC3D[17] = 4      # car
LIDARSEG_TO_OCC3D[18] = 5      # construction_vehicle
LIDARSEG_TO_OCC3D[21] = 6      # motorcycle
LIDARSEG_TO_OCC3D[22] = 9      # trailer
LIDARSEG_TO_OCC3D[23] = 10     # truck
LIDARSEG_TO_OCC3D[24] = 11     # driveable_surface
LIDARSEG_TO_OCC3D[25] = 12     # other_flat
LIDARSEG_TO_OCC3D[26] = 13     # sidewalk
LIDARSEG_TO_OCC3D[27] = 14     # terrain
LIDARSEG_TO_OCC3D[28] = 15     # manmade
LIDARSEG_TO_OCC3D[30] = 16     # vegetation

# NuScenes lidarseg class names (0-31)
LIDARSEG_NAMES = [
    "noise",                    # 0
    "animal",                   # 1
    "human.pedestrian.adult",   # 2
    "human.pedestrian.child",   # 3
    "human.pedestrian.construction_worker",  # 4
    "human.pedestrian.personal_mobility",    # 5
    "human.pedestrian.police_officer",       # 6
    "human.pedestrian.stroller",             # 7
    "human.pedestrian.wheelchair",           # 8
    "movable_object.barrier",                # 9
    "movable_object.debris",                 # 10
    "movable_object.pushable_pullable",      # 11
    "movable_object.trafficcone",            # 12
    "static_object.bicycle_rack",            # 13
    "vehicle.bicycle",                       # 14
    "vehicle.bus.bendy",                     # 15
    "vehicle.bus.rigid",                     # 16
    "vehicle.car",                           # 17
    "vehicle.construction",                  # 18
    "vehicle.emergency.ambulance",           # 19
    "vehicle.emergency.police",              # 20
    "vehicle.motorcycle",                    # 21
    "vehicle.trailer",                       # 22
    "vehicle.truck",                         # 23
    "flat.driveable_surface",                # 24
    "flat.other",                            # 25
    "flat.sidewalk",                         # 26
    "flat.terrain",                          # 27
    "static.manmade",                        # 28
    "static.other",                          # 29
    "static.vegetation",                     # 30
    "vehicle.ego"                            # 31
]

# Occ3D class names (0-16)
OCC3D_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]

print("="*70)
print("Current Lidarseg -> Occ3D Mapping")
print("="*70)
print(f"{'Lidarseg':>3} | {'Lidarseg Name':>35} | {'Occ3D':>5} | {'Occ3D Name':>20}")
print("-"*70)
for i in range(32):
    occ_idx = LIDARSEG_TO_OCC3D[i]
    occ_name = OCC3D_NAMES[occ_idx] if occ_idx < 17 else 'free'
    print(f"{i:3d} | {LIDARSEG_NAMES[i]:>35} | {occ_idx:5d} | {occ_name:>20}")

print("\n" + "="*70)
print("Label Match Analysis at Same Locations")
print("="*70)

valid_mask = (mask == 1)
pred_valid = pred[valid_mask]
gt_valid = gt[valid_mask]

# Check where they differ
diff_mask = (pred_valid != gt_valid) & (gt_valid != 17)
diff_pred = pred_valid[diff_mask]
diff_gt = gt_valid[diff_mask]

print(f"\n[Mismatch analysis] {diff_mask.sum()} mismatches at GT occupied locations")
print(f"\nTop confusion pairs (GT -> Pred):")
from collections import Counter
pairs = Counter(zip(diff_gt, diff_pred))
for (gt_lbl, pred_lbl), cnt in pairs.most_common(20):
    gt_name = OCC3D_NAMES[gt_lbl] if gt_lbl < 17 else 'free'
    pred_name = OCC3D_NAMES[pred_lbl] if pred_lbl < 17 else 'free'
    print(f"  GT={gt_lbl:2d}({gt_name:>15}) -> Pred={pred_lbl:2d}({pred_name:>15}): {cnt}")
