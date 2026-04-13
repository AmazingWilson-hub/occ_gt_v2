
import numpy as np

gt_path = "/home/t113c52027/t113c52027/occ_gt_v2/data/nuscenes_occ/gts/scene-0061/ca9a282c9e77460f8360f564131a8af5/labels.npz"
pred_path = "/home/t113c52027/t113c52027/occ_gt_v2/presentation_exporter/output_demo/scene-0061/000_ca9a282c9e77460f8360f564131a8af5/OCCUPANCY.npz"

print("="*60)
print("Label Value Comparison")
print("="*60)

# GT
gt_data = np.load(gt_path)
gt_sem = gt_data['semantics']
gt_mask = gt_data['mask_lidar']

print("\n[GT] Unique labels in entire volume:")
print(np.unique(gt_sem))

print("\n[GT] Unique labels in valid region (mask_lidar==1):")
gt_valid = gt_sem[gt_mask == 1]
print(np.unique(gt_valid))

# Pred
pred_data = np.load(pred_path)
pred_sem = pred_data['semantics']

print("\n[Pred] Unique labels (raw NuScenes lidarseg):")
print(np.unique(pred_sem))

# NuScenes lidarseg class names (0-31)
LIDARSEG_NAMES = [
    "noise",           # 0
    "animal",          # 1
    "human.pedestrian.adult",     # 2
    "human.pedestrian.child",     # 3
    "human.pedestrian.construction_worker",  # 4
    "human.pedestrian.personal_mobility",    # 5
    "human.pedestrian.police_officer",       # 6
    "human.pedestrian.stroller",             # 7
    "human.pedestrian.wheelchair",           # 8
    "barrier",                   # 9
    "debris",                    # 10
    "movable_object.pushable_pullable",      # 11
    "traffic_cone",              # 12
    "pole",            # 13
    "bicycle",                   # 14
    "bus.bendy",                 # 15
    "bus.rigid",                 # 16
    "car",                       # 17
    "construction_vehicle",      # 18
    "emergency.ambulance",       # 19
    "emergency.police",          # 20
    "motorcycle",                # 21
    "trailer",                   # 22
    "truck",                     # 23
    "driveable_surface",         # 24
    "other_flat",                # 25
    "sidewalk",                  # 26
    "terrain",                   # 27
    "manmade",                   # 28
    "pole",            # 29
    "vegetation",                # 30
    "ego"                        # 31
]

print("\n[Reference] NuScenes lidarseg (0-31):")
for i, name in enumerate(LIDARSEG_NAMES):
    print(f"  {i:2d}: {name}")
