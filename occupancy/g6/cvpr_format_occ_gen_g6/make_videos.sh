#!/bin/bash
# Generate comparison videos for all G6 scenes
set -e
cd "$(dirname "$0")/.."

OUT_DIR="tools/compare_video/video_out"
OCC_ROOT="cvpr_format_occ_gen_g6/output"
CAM_ROOT="data/g6"

SCENES=(
    "citystreet_sunny_day_2026-02-03-15-17-34"
    "citystreet_sunny_day_2026-02-03-16-51-01"
    "citystreet_sunny_day_2026-02-03-17-00-10"
    "citystreet_sunny_day_2026-02-03-17-03-09"
    "citystreet_sunny_day_2026-03-02-14-13-43"
    "citystreet_sunny_day_2026-03-02-14-13-58"
    "citystreet_sunny_day_2026-03-02-14-14-33"
    "citystreet_sunny_day_2026-03-02-14-16-53"
    "citystreet_sunny_day_2026-03-02-14-19-39"
    "citystreet_sunny_day_2026-03-02-14-28-54"
    "citystreet_sunny_day_2026-03-02-14-33-32"
    "citystreet_sunny_day_2026-03-02-14-34-40"
    "citystreet_sunny_day_2026-03-02-14-35-14"
    "citystreet_sunny_day_2026-03-02-14-36-02"
    "citystreet_sunny_day_2026-03-02-15-06-09"
    "citystreet_sunny_day_2026-03-02-15-06-36"
    "citystreet_sunny_day_2026-03-02-15-14-03"
    "citystreet_rainy_day_2026-03-09-15-54-19"
    "citystreet_sunny_day_2026-03-09-10-47-58"
    "countryside_rainy_day_2026-03-09-15-29-09"
)

for scene in "${SCENES[@]}"; do
    if [ -f "$OUT_DIR/g6_${scene}.mp4" ]; then
        echo "  [SKIP] $scene"
        continue
    fi
    echo "Rendering: $scene"

    # Use seg as bev_dir if available, else heuristic
    if [ -d "$OCC_ROOT/$scene/seg" ] && [ "$(ls $OCC_ROOT/$scene/seg/ | wc -l)" -gt 0 ]; then
        bev_dir="$OCC_ROOT/$scene/seg"
        dirs="$OCC_ROOT/$scene/raw $OCC_ROOT/$scene/heuristic $OCC_ROOT/$scene/seg"
        labels="Raw Heuristic Semantic"
    else
        bev_dir="$OCC_ROOT/$scene/heuristic"
        dirs="$OCC_ROOT/$scene/raw $OCC_ROOT/$scene/heuristic"
        labels="Raw Heuristic"
    fi

    # 新場景用 images（6cam），舊場景用 image（single）
    if [ -d "$CAM_ROOT/$scene/images" ]; then
        cam_dir="$CAM_ROOT/$scene/images"
    else
        cam_dir="$CAM_ROOT/$scene/image"
    fi

    # 自動偵測副檔名（jpg 或 png）
    if ls "$cam_dir"/main/*.png "$cam_dir"/*.png "$cam_dir"/main/*.PNG 2>/dev/null | head -1 | grep -q png; then
        cam_ext="--cam_ext .png"
    else
        cam_ext=""
    fi

    python3 tools/compare_video/run.py \
        --dirs $dirs \
        --labels $labels \
        --bev_dir "$bev_dir" \
        --cam_dir "$cam_dir" \
        $cam_ext \
        --grid_z 21 \
        --voxel_style shaded \
        --out "$OUT_DIR/g6_${scene}.mp4" \
        --z_offset -3.0

    echo "  Saved: $OUT_DIR/g6_${scene}.mp4"
done

echo ""
echo "All G6 videos complete."
