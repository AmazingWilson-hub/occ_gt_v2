#!/bin/bash
# 解壓新 G6 場景的 image_label.zip 到同名目錄
set -e

G6_ROOT="/data2/t113c52027/occ_gt_v2/data/g6"

SCENES=(
    "citystreet_sunny_day_2026-03-02-14-19-39"
    "citystreet_sunny_day_2026-03-02-14-28-54"
    "citystreet_sunny_day_2026-03-02-14-33-32"
)

for scene in "${SCENES[@]}"; do
    zip_path="$G6_ROOT/$scene/image_label.zip"
    out_dir="$G6_ROOT/$scene/image_label"

    if [ ! -f "$zip_path" ]; then
        echo "[SKIP] $scene: image_label.zip not found"
        continue
    fi

    if [ -d "$out_dir" ] && [ "$(ls "$out_dir" | wc -l)" -gt 0 ]; then
        echo "[SKIP] $scene: image_label/ already exists ($(ls "$out_dir" | wc -l) files)"
        continue
    fi

    echo "Unzipping: $scene"
    mkdir -p "$out_dir"
    unzip -q "$zip_path" -d "$out_dir"
    echo "  Done: $(find "$out_dir" -name '*.npz' | wc -l) npz files"
done

echo ""
echo "All done."
