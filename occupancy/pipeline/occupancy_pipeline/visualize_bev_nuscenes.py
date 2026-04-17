import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse

# NuScenes Class Color Map (RGB 0-255) based on vis_pts.py
# Order matches the dictionary keys in vis_pts.py
NUSC_COLORS = np.array([
    [0, 0, 0],             # 0: noise
    [70, 130, 180],        # 1: animal
    [0, 0, 230],           # 2: human.pedestrian.adult
    [0, 0, 230],           # 3: human.pedestrian.child
    [0, 0, 230],           # 4: human.pedestrian.construction_worker
    [0, 0, 230],           # 5: human.pedestrian.personal_mobility
    [0, 0, 230],           # 6: human.pedestrian.police_officer
    [0, 0, 230],           # 7: human.pedestrian.stroller
    [0, 0, 230],           # 8: human.pedestrian.wheelchair
    [112, 128, 144],       # 9: movable_object.barrier
    [112, 128, 144],       # 10: movable_object.debris
    [112, 128, 144],       # 11: movable_object.pushable_pullable
    [112, 128, 144],       # 12: movable_object.trafficcone
    [188, 143, 143],       # 13: static_object.bicycle_rack
    [220, 20, 60],         # 14: vehicle.bicycle
    [255, 158, 0],         # 15: vehicle.bus.bendy
    [255, 158, 0],         # 16: vehicle.bus.rigid
    [255, 158, 0],         # 17: vehicle.car
    [255, 158, 0],         # 18: vehicle.construction
    [255, 158, 0],         # 19: vehicle.emergency.ambulance
    [255, 158, 0],         # 20: vehicle.emergency.police
    [255, 158, 0],         # 21: vehicle.motorcycle
    [255, 158, 0],         # 22: vehicle.trailer
    [255, 158, 0],         # 23: vehicle.truck
    [0, 207, 191],         # 24: flat.driveable_surface
    [0, 207, 191],         # 25: flat.other
    [75, 0, 75],           # 26: flat.sidewalk
    [0, 207, 191],         # 27: flat.terrain
    [222, 184, 135],       # 28: static.manmade
    [0, 207, 191],         # 29: static.other
    [0, 175, 0],           # 30: static.vegetation
    [255, 240, 245],       # 31: vehicle.ego
])

def visualize_bev(npz_path, output_path):
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    if 'semantics' in data:
        grid = data['semantics']
    elif 'arr_0' in data:
        grid = data['arr_0']
    else:
        print("Error: Could not find 'semantics' in npz file.")
        return

    print(f"Grid shape: {grid.shape}")
    
    # Project to BEV (Top-down view)
    nx, ny, nz = grid.shape
    bev_map = np.zeros((nx, ny), dtype=grid.dtype)
    
    # Iterate from bottom to top
    for z in range(nz):
        slice_z = grid[:, :, z]
        mask = slice_z > 0
        bev_map[mask] = slice_z[mask]
        
    print("Applying NuScenes color map...")
    
    # Initialize RGB image
    bev_rgb = np.zeros((nx, ny, 3), dtype=np.uint8)
    
    # Handle labels
    # We clip labels to be within 0-31. Any label > 31 is treated as 0 (noise/black)
    # or we can check unique labels first.
    
    unique_labels = np.unique(bev_map)
    print(f"Unique labels found: {unique_labels}")
    
    for label in unique_labels:
        if label == 0:
            continue # Background remains black
            
        if label < 0 or label >= len(NUSC_COLORS):
            print(f"Warning: Label {label} is out of range (0-31). Mapping to Black.")
            continue
            
        # Apply color
        color = NUSC_COLORS[int(label)]
        mask = (bev_map == label)
        bev_rgb[mask] = color
        
    print(f"Saving to {output_path}...")
    
    # Rotate 90 deg counter-clockwise to match typical NuScenes orientation (X forward, Y left)
    # The grid is usually (X, Y, Z). 
    # Image (row, col) maps to (X, Y) usually.
    # We'll just save it directly first.
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(bev_rgb, (1, 0, 2)), origin='lower') # Transpose to put X up? 
    # If X is forward (up) and Y is left.
    # Array (X, Y) -> Imshow (Row, Col). 
    # If we want X up, we need X to be rows (0) and inverted?
    # Usually standard is enough.
    
    plt.axis('off')
    plt.title("BEV Occupancy (NuScenes Colors)")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    visualize_bev("sample_occupancy_real.npz", "sample_occupancy_bev_nuscenes.png")
