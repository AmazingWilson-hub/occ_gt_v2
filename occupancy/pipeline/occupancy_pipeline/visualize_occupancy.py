import numpy as np
import open3d as o3d
import argparse
import os
import sys

import matplotlib.pyplot as plt

# NuScenes Official Color Map (0-31)
# Normalized to [0, 1]
NUSC_COLORS = np.array([
    [0, 0, 0],             # 0: noise
    [70, 130, 180],        # 1: animal
    [0, 0, 230],           # 2: human.pedestrian.adult
    [135, 206, 235],       # 3: human.pedestrian.child
    [100, 149, 237],       # 4: human.pedestrian.construction_worker
    [219, 112, 147],       # 5: human.pedestrian.personal_mobility
    [0, 0, 128],           # 6: human.pedestrian.police_officer
    [240, 128, 128],       # 7: human.pedestrian.stroller
    [138, 43, 226],        # 8: human.pedestrian.wheelchair
    [112, 128, 144],       # 9: movable_object.barrier
    [210, 105, 30],        # 10: movable_object.debris
    [105, 105, 105],       # 11: movable_object.pushable_pullable
    [47, 79, 79],          # 12: movable_object.trafficcone
    [188, 143, 143],       # 13: static_object.bicycle_rack
    [220, 20, 60],         # 14: vehicle.bicycle
    [255, 127, 80],        # 15: vehicle.bus.bendy
    [255, 69, 0],          # 16: vehicle.bus.rigid
    [255, 158, 0],         # 17: vehicle.car
    [233, 150, 70],        # 18: vehicle.construction
    [255, 83, 0],          # 19: vehicle.emergency.ambulance
    [255, 215, 0],         # 20: vehicle.emergency.police
    [255, 61, 99],         # 21: vehicle.motorcycle
    [255, 140, 0],         # 22: vehicle.trailer
    [255, 99, 71],         # 23: vehicle.truck
    [0, 207, 191],         # 24: flat.driveable_surface
    [175, 0, 75],          # 25: flat.other
    [75, 0, 75],           # 26: flat.sidewalk
    [112, 180, 60],        # 27: flat.terrain
    [222, 184, 135],       # 28: static.manmade
    [255, 228, 196],       # 29: static.other
    [0, 175, 0],           # 30: static.vegetation
    [255, 240, 245],       # 31: vehicle.ego
]) / 255.0

def render_bev(indices, semantics, save_path):
    print("Generating BEV image...")
    # Grid size: 120m / 0.2m = 600
    grid_size = 600
    bev_map = np.ones((grid_size, grid_size, 3)) # White background
    
    # Sort by height (Z) so higher points overwrite lower ones (simple occlusion)
    # indices is [N, 3] (x, y, z)
    sorted_idx = np.argsort(indices[:, 2])
    indices = indices[sorted_idx]
    semantics = semantics[sorted_idx]
    
    # Color palette
    label_colors = NUSC_COLORS
    
    count = 0
    for idx, label in zip(indices, semantics):
        x, y = idx[0], idx[1]
        # Check bounds
        if 0 <= x < grid_size and 0 <= y < grid_size:
            lbl_idx = int(label)
            color = label_colors[lbl_idx] if 0 <= lbl_idx < len(label_colors) else [0,0,0]
            # Image y is inverted relative to cartesian y usually, but let's just map direct
            # To make it look like a map, usually x is right, y is up. 
            # Array is [row, col] -> [y, x]. Let's create [grid_size-1-y, x]
            bev_map[grid_size - 1 - y, x] = color
            count += 1
            
    print(f"Projected {count} voxels to BEV.")
    plt.imsave(save_path, bev_map)
    print(f"Saved BEV image to {save_path}")

def visualize_npz(file_path, save_ply=None, save_img=None):
    print(f"Loading {file_path}...")
    try:
        data = np.load(file_path)
        indices = data['indices']
        semantics = data['semantics']
    except Exception as e:
        print(f"Error loading NPZ: {e}")
        return
        
    if save_img:
        render_bev(indices, semantics, save_img)
        # If user only asks for image, we can return or continue. 
        # Continue to PLY saving if requested.
        if not save_ply and 'save_ply' not in sys.argv: # Heuristic
             pass 

    # Reconstruction parameters (Must match converter)
    min_bound = np.array([-60.0, -60.0, -5.0])
    voxel_size = 0.2
    
    # Convert indices to point centroids
    xyz = (indices * voxel_size) + min_bound + (voxel_size / 2.0)
    
    print(f"Loaded {len(xyz)} voxels.")
    
    # Colorize
    colors = np.zeros((len(semantics), 3))
    label_colors = NUSC_COLORS
    
    for i in range(len(semantics)):
        lbl = int(semantics[i])
        if 0 <= lbl < len(label_colors):
            colors[i] = label_colors[lbl]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if save_ply:
        print(f"Saving PLY to {save_ply}...")
        o3d.io.write_point_cloud(save_ply, pcd)
        print("Done.")
        return
    
    if not save_img:    
        print("Visualizing (Window mode)...")
        o3d.visualization.draw_geometries([pcd])

def visualize_pcd_bin(file_path, save_ply=None, use_labels=False):
    print(f"Loading {file_path}...")
    # Load raw binary data
    # format is float16 as per data_converter.py: new_points = new_points.astype(np.float16)
    data = np.fromfile(file_path, dtype=np.float16)
    
    # Reshape: (N, 5) -> x, y, z, intensity, label
    try:
        points = data.reshape(-1, 5)
    except ValueError:
        print(f"Error: Could not reshape data of size {data.size} into (N, 5)")
        print("Trying to guess shape... maybe 4 or 6 cols?")
        return

    xyz = points[:, :3]
    intensity = points[:, 3]
    labels = points[:, 4]
    
    print(f"Loaded {len(points)} points.")
    print(f"Coordinate range: X: {xyz[:,0].min():.2f}~{xyz[:,0].max():.2f}, Y: {xyz[:,1].min():.2f}~{xyz[:,1].max():.2f}, Z: {xyz[:,2].min():.2f}~{xyz[:,2].max():.2f}")
    
    # Check if labels are meaningful
    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")
    
    colors = np.zeros((len(labels), 3))
    
    # Heuristic: If we only have 1 label (dummy) or user requests height, color by height
    # BUT if user explicitly asks for labels, obey them.
    use_height = (len(unique_labels) <= 1) and (not use_labels)
    
    if use_height:
        print("Note: Detected uniform dummy labels (or missing implementation). Switching to HEIGHT-based coloring.")
        z = xyz[:, 2]
        # Normalize Z to 0-1 for coloring
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        # Simple heatmap (Blue->Red)
        import matplotlib.cm
        cmap = matplotlib.cm.get_cmap('jet')
        colors = cmap(z_norm)[:, :3]
    else:
        # Colorize based on labels
        max_label = int(labels.max())
        print(f"Coloring by Semantic Label (Max: {max_label})")
        
        # Simple consistent coloring (random but deterministic)
        np.random.seed(42)
        label_colors = np.random.rand(max_label + 1, 3)
        
        # Override for specific classes if known
        # 24 = driveable surface (gray)
        if max_label >= 24:
            label_colors[24] = [0.5, 0.5, 0.5] 
        # 17 = car (blue)
        if max_label >= 17:
             label_colors[17] = [0.0, 0.0, 1.0]

        for i in range(len(labels)):
            lbl = int(labels[i])
            if lbl <= max_label:
                colors[i] = label_colors[lbl]
            
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if save_ply:
        print(f"Saving PLY to {save_ply}...")
        o3d.io.write_point_cloud(save_ply, pcd)
        print("Done.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Occupancy Viz", width=1280, height=720)
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0, 0, 0])
    
    print("Visualizing... Close window to exit.")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize .pcd.bin occupancy files")
    parser.add_argument("file", help="Path to .pcd.bin file")
    parser.add_argument("--save_ply", help="Path to save cleaned PLY file (optional)", default=None)
    parser.add_argument("--save_img", help="Path to save BEV image (optional, NPZ only)", default=None)
    parser.add_argument("--use_labels", help="Force use of semantic labels even if uniform", action="store_true")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
    elif args.file.endswith('.npz'):
        visualize_npz(args.file, save_ply=args.save_ply, save_img=args.save_img)
    else:
        visualize_pcd_bin(args.file, save_ply=args.save_ply, use_labels=args.use_labels)
