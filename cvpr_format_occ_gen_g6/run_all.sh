#!/bin/bash
# Run occupancy generation for all G6 scenes
# Usage: bash run_all.sh [--mode all|raw|heuristic|semantic] [--num_workers N]

set -e
cd "$(dirname "$0")"

MODE=${1:-all}
NUM_WORKERS=${2:-8}

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
    echo "========================================"
    echo "Processing: $scene"
    echo "========================================"

    # scenes without semantic data: skip seg mode
    # 檢查語意點雲是否完整（幀數需與 VLS128 一致）
    no_seg=0
    scene_path="../data/g6/$scene"
    vls_count=$(ls "$scene_path/VLS128_pcd/" 2>/dev/null | wc -l)
    if [ "$vls_count" -eq 0 ]; then
        vls_count=$(ls "$scene_path/VLS128_pcdnpy/" 2>/dev/null | wc -l)
    fi
    sem_count=$(ls "$scene_path/result_depth_filtered_v2/" 2>/dev/null | wc -l)
    if [ "$sem_count" -eq 0 ]; then
        sem_count=$(ls "$scene_path/result_depth_filtered/" 2>/dev/null | wc -l)
    fi
    if [ "$sem_count" -eq 0 ]; then
        sem_count=$(ls "$scene_path/colored_360_pcd_filter/" 2>/dev/null | wc -l)
    fi
    if [ "$vls_count" -eq 0 ] && [ -d "$scene_path/VLS128_pcdnpy" ]; then
        vls_count=$(ls "$scene_path/VLS128_pcdnpy/" 2>/dev/null | wc -l)
    fi
    if [ "$sem_count" -lt "$vls_count" ]; then
        no_seg=1
        echo "  [NOTE] Incomplete semantic ($sem_count/$vls_count frames), skipping seg" | tee -a DATA_NOTES_incomplete.log
    fi

    if [[ $no_seg -eq 1 && "$MODE" == "semantic" ]]; then
        echo "  [SKIP] No semantic data for $scene"
        continue
    fi

    if [ "$MODE" == "all" ]; then
        if [[ $no_seg -eq 1 ]]; then
            run_mode="raw heuristic"
        else
            run_mode="raw heuristic semantic"
        fi
    else
        run_mode="$MODE"
    fi

    for m in $run_mode; do
        out_dir="output/$scene/$m"
        # semantic mode outputs to "seg" directory
        if [ "$m" == "semantic" ]; then out_dir="output/$scene/seg"; fi
        if [ -d "$out_dir" ] && [ "$(ls "$out_dir" | wc -l)" -gt 0 ]; then
            echo "  [SKIP] $m already exists ($(ls "$out_dir" | wc -l) frames)"
            continue
        fi
        python3 generate.py --scene "$scene" --mode "$m" --num_workers "$NUM_WORKERS"
    done

    echo "  Done: $scene"
done

echo ""
echo "All G6 scenes complete."
