import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualize_ply_raw_colors(ply_path, output_path):
    print(f"Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) # (N, 3), range 0-1
    
    print(f"Points: {points.shape}")
    print(f"Colors sample (RGB): \n{colors[:5]}")
    
    # Define Grid
    min_bound = np.array([-51.2, -51.2, -5.0])
    max_bound = np.array([51.2, 51.2, 3.0])
    voxel_size = 0.2
    grid_shape = ((max_bound - min_bound) / voxel_size).astype(int)
    
    # Initialize BEV image (White background for better visibility of dark points, or Black)
    # PLY usually viewed on black background.
    bev_img = np.zeros((grid_shape[0], grid_shape[1], 3), dtype=np.float32)
    
    # To handle overlapping points, we can use a depth buffer or just "highest point wins"
    # Let's use Z-buffer approach
    z_buffer = np.full((grid_shape[0], grid_shape[1]), -np.inf)
    
    print("Projecting raw colors to BEV...")
    
    # Calculate indices
    indices = ((points - min_bound) / voxel_size).astype(int)
    
    # Filter valid
    valid_mask = (
        (indices[:, 0] >= 0) & (indices[:, 0] < grid_shape[0]) &
        (indices[:, 1] >= 0) & (indices[:, 1] < grid_shape[1])
    )
    
    valid_points = points[valid_mask]
    valid_indices = indices[valid_mask]
    valid_colors = colors[valid_mask]
    
    # Simple loop or vectorized sort to handle Z-ordering
    # Sort by Z height ascending, so higher points overwrite lower points
    sort_idx = np.argsort(valid_points[:, 2])
    
    valid_indices = valid_indices[sort_idx]
    valid_colors = valid_colors[sort_idx]
    
    # Assign colors
    # Because we sorted by Z, the last assignment for a pixel is the highest point
    bev_img[valid_indices[:, 0], valid_indices[:, 1]] = valid_colors
    
    print(f"Saving to {output_path}...")
    
    plt.figure(figsize=(12, 12))
    # Rotate to standard NuScenes orientation (X up, Y left)
    # Transpose (X,Y) -> (Y,X) then usually flip to match map
    # Let's just use origin='lower' and transpose to get X vertical
    plt.imshow(np.transpose(bev_img, (1, 0, 2)), origin='lower')
    plt.axis('off')
    plt.title(f"Raw PLY Color Projection\n(R=Intensity, G=Label, B=0)")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    visualize_ply_raw_colors("sample_occupancy_real.ply", "sample_occupancy_ply_raw.png")
