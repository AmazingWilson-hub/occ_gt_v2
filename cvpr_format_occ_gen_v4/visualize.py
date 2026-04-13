import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

# Occ3D Class Color Map (RGB 0-255)
# 17 semantic classes + 1 free space
OCC3D_COLORS = np.array([
    [0, 0, 0],             # 0: noise / others
    [112, 128, 144],       # 1: barrier
    [220, 20, 60],         # 2: bicycle
    [255, 158, 0],         # 3: bus
    [255, 158, 0],         # 4: car
    [255, 158, 0],         # 5: construction_vehicle
    [255, 158, 0],         # 6: motorcycle
    [0, 0, 230],           # 7: pedestrian
    [112, 128, 144],       # 8: traffic_cone
    [255, 158, 0],         # 9: trailer
    [255, 158, 0],         # 10: truck
    [0, 207, 191],         # 11: driveable_surface
    [0, 207, 191],         # 12: other_flat
    [75, 0, 75],           # 13: sidewalk
    [0, 207, 191],         # 14: terrain
    [222, 184, 135],       # 15: manmade
    [0, 175, 0],           # 16: vegetation
    [255, 255, 255],       # 17: free_space
])

def visualize_bev(npz_path, output_path):
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    grid = data['semantics']
    
    nx, ny, nz = grid.shape
    bev_map = np.ones((nx, ny), dtype=np.uint8) * 17
    
    # Iterate from bottom to top to project
    for z in range(nz):
        slice_z = grid[:, :, z]
        mask = slice_z != 17
        bev_map[mask] = slice_z[mask]
        
    bev_rgb = np.zeros((nx, ny, 3), dtype=np.uint8)
    
    unique_labels = np.unique(bev_map)
    for label in unique_labels:
        if label < 0 or label > 17:
            continue
        color = OCC3D_COLORS[int(label)]
        bev_rgb[bev_map == label] = color
        
    plt.figure(figsize=(10, 10))
    # Standard orientation: X forward (up), Y left (left)
    plt.imshow(np.transpose(bev_rgb, (1, 0, 2)), origin='lower')
    plt.axis('off')
    plt.title(f"BEV Occupancy (Occ3D Colors) - {os.path.basename(os.path.dirname(npz_path))}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', default=os.path.join(os.path.dirname(__file__), 'output/kiss_slam/scene-0061'))
    parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(__file__), 'vis_out'))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find all npz files
    tokens = os.listdir(args.pred_dir)
    print(f"Found {len(tokens)} frames to visualize.")
    
    for token in tokens:
        npz_path = os.path.join(args.pred_dir, token, 'labels.npz')
        if not os.path.exists(npz_path):
            continue
        
        out_img_path = os.path.join(args.out_dir, f"{token}_bev.png")
        visualize_bev(npz_path, out_img_path)

if __name__ == '__main__':
    main()
