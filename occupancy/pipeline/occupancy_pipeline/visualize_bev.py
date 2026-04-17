import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse

def visualize_bev(npz_path, output_path):
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    if 'semantics' in data:
        grid = data['semantics']
    elif 'arr_0' in data:
        grid = data['arr_0']
    else:
        print("Error: Could not find 'semantics' or 'arr_0' in npz file.")
        print(f"Keys found: {list(data.keys())}")
        return

    print(f"Grid shape: {grid.shape}")
    
    # Grid dimensions: (X, Y, Z) -> (512, 512, 40)
    # We want to project to BEV (X, Y).
    # Strategy: For each (x, y), take the label of the highest occupied voxel (largest Z).
    
    nx, ny, nz = grid.shape
    bev_map = np.zeros((nx, ny), dtype=grid.dtype)
    
    # Iterate from bottom to top, so higher objects overwrite lower ones
    print("Projecting to BEV...")
    for z in range(nz):
        # specific Z slice
        slice_z = grid[:, :, z]
        # update mask: where current slice has non-zero labels
        mask = slice_z > 0
        bev_map[mask] = slice_z[mask]
        
    print("Generating image...")
    # Create a colormap
    # Get unique labels
    unique_labels = np.unique(bev_map)
    print(f"Unique labels in BEV: {unique_labels}")
    
    # Normalize labels for coloring
    # We'll use a "jet" colormap but handle background (0) separately
    
    plt.figure(figsize=(10, 10))
    
    # Mask out background (0)
    masked_bev = np.ma.masked_where(bev_map == 0, bev_map)
    
    # Plot
    # Rotated 90 degrees to match typical map orientation if needed, 
    # but standard array plotting is usually fine. 
    # imshow origin is typically 'upper', but lidar data often has x forward, y left.
    # We'll just plot standard matrix for now.
    
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    plt.imshow(masked_bev, cmap=cmap, origin='lower')
    
    plt.colorbar(label='Semantic Class')
    plt.title(f"BEV Occupancy Visualization\nSource: {npz_path}")
    plt.axis('off')
    
    print(f"Saving to {output_path}...")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    visualize_bev("sample_occupancy_real.npz", "sample_occupancy_bev.png")
