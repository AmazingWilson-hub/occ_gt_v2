#!/bin/bash
# run_exporter.sh

# Stop on error
set -e

# Configuration
VERSION="v1.0-trainval"
DATAROOT="../data/nuscenes_occ"
OUTPUT_DIR="output_demo"
SCENE_NAME="scene-0061" # Choose a scene from mini

# Get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Resolve Dataroot relative to script
DATAROOT="$DIR/../data/nuscenes_occ"

# Run
echo "Running Presentation Exporter..."
echo "Scene: $SCENE_NAME"
echo "Output: $OUTPUT_DIR"

cd "$DIR" # Go into presentation_exporter dir to make imports work relative to it

python3 generate_demo_scene.py \
    --version "$VERSION" \
    --dataroot "$DATAROOT" \
    --out_root "$OUTPUT_DIR" \
    --scene_name "$SCENE_NAME"

echo "Done! Check $DIR/$OUTPUT_DIR"
