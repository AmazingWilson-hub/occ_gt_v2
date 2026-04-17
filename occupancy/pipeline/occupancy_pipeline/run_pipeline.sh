#!/bin/bash
set -e

# Default Paths (Adjust as needed)
DATAROOT="../data/nuscenes_occ/" 
OUTPUT_BIN="output_bin/"
OUTPUT_NPZ="output_npz/"
VERSION="v1.0-trainval" 

echo "=========================================="
echo "Starting Occupancy Generation Pipeline"
echo "Data Root: $DATAROOT"
echo "Version:   $VERSION"
echo "=========================================="

# 1. Generate Raw .pcd.bin Files
echo "[Step 1] Generating raw .pcd.bin point clouds..."
# Note: Using python3 -u for unbuffered output to see progress
python3 data_converter.py --dataroot "$DATAROOT" --save_path "$OUTPUT_BIN" --version "$VERSION"

# 2. Convert to CVPR Challenge Format (.npz)
echo "[Step 2] Converting to .npz (Sparse Voxel Grid)..."
# Since convert_to_npz.py saves in-place by default if out_dir isn't supported, 
# let's modify the call or create output dir first.
# Wait, my convert_to_npz.py currently saves next to input file if --out is not provided.
# I will update it to support output directory or move files after.
# For now, let's assume in-place generation and then move, or just keep them there.
# But better: convert_to_npz.py input is directory. 
# My script convert_to_npz.py currently writes .npz next to .bin if directory input is used.
# Let's keep it simple.

python3 convert_to_npz.py "$OUTPUT_BIN" --num_workers 16

# Optional: Move to separate folder if needed, but currently they sit together.
mkdir -p "$OUTPUT_NPZ"
mv "$OUTPUT_BIN"/*.npz "$OUTPUT_NPZ"/ 2>/dev/null || true

echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "Outputs:"
echo "  Raw BIN: $OUTPUT_BIN"
echo "  Final NPZ: $OUTPUT_NPZ"
echo "=========================================="
