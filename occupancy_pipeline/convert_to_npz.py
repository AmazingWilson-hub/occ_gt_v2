import numpy as np
import argparse
import os
import sys

def convert_bin_to_npz(file_path, save_path=None):
    print(f"Processing {file_path}...")
    try:
        data = np.fromfile(file_path, dtype=np.float16)
        points = data.reshape(-1, 5) # x, y, z, intensity, label
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Define bounds and voxel size (Expanded to match original script / data range)
    # Range: [-60, 60] X/Y, [-5, 11] Z to include all points
    min_bound = np.array([-60.0, -60.0, -5.0])
    max_bound = np.array([60.0, 60.0, 11.0])
    voxel_size = 0.2

    xyz = points[:, :3]
    labels = points[:, 4]

    # Filter points within bounds
    mask = (xyz[:, 0] >= min_bound[0]) & (xyz[:, 0] < max_bound[0]) & \
           (xyz[:, 1] >= min_bound[1]) & (xyz[:, 1] < max_bound[1]) & \
           (xyz[:, 2] >= min_bound[2]) & (xyz[:, 2] < max_bound[2])
    
    valid_xyz = xyz[mask]
    valid_labels = labels[mask].astype(int)
    
    if len(valid_xyz) == 0:
        print("Warning: No points within defined bounds.")
        return

    # Voxelize to integer indices
    indices = ((valid_xyz - min_bound) / voxel_size).astype(int)
    
    # Determine save path
    if save_path is None:
        save_path = file_path.replace('.pcd.bin', '.npz').replace('.bin', '.npz')
    
    print(f"Saving to {save_path}...")
    print(f"  Shape: indices={indices.shape}, semantics={valid_labels.shape}")
    
    np.savez_compressed(save_path, indices=indices, semantics=valid_labels)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pcd.bin to .npz (CVPR2023 format)")
    parser.add_argument("input", help="Path to .pcd.bin file OR directory containing them")
    parser.add_argument("--out", help="Output filename (optional, only for single file)", default=None)
    parser.add_argument("--num_workers", help="Number of parallel workers", type=int, default=8)
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        convert_bin_to_npz(args.input, args.out)
    elif os.path.isdir(args.input):
        from concurrent.futures import ProcessPoolExecutor
        import glob
        
        files = glob.glob(os.path.join(args.input, "**/*.pcd.bin"), recursive=True)
        print(f"Found {len(files)} .pcd.bin files in {args.input}")
        
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            list(executor.map(convert_bin_to_npz, files))
    else:
        print(f"Input not found: {args.input}")
        sys.exit(1)
