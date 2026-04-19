# Paper Figures

產生論文 Section 3.5 的三組 occupancy 圖片。

## 產生的檔案

| 腳本 | 輸出檔案 | 對應論文圖 |
|------|----------|-----------|
| `fig1_occ_bev.py` | `occ_bev_visualization.png` | Fig. occupancy BEV（17 類顏色俯視圖）|
| `fig2_sweep_comparison.py` | `occ_sweep_1.png` / `occ_sweep_40.png` | Fig. 多幀累積效果（1 vs 40 幀）|
| `fig3_pose_comparison.py` | `occ_pose_gt.png` / `occ_pose_kiss.png` | Fig. Pose 品質比較（GT vs KISS-SLAM）|

## 執行方式

```bash
cd /data2/t113c52027/occ_gt_v2/tools/paper_figures

# Fig 1: Occupancy BEV 可視化
python3 fig1_occ_bev.py \
    --pred_dir ../../occupancy/nuscenes/v3/output/gt_pose/scene-0061 \
    --out paper_out/occ_bev_visualization.png

# Fig 2: 多幀累積比較（1 sweep vs 40 sweeps）
python3 fig2_sweep_comparison.py \
    --dir_1sweep  ../../occupancy/nuscenes/v1/output/scene-0061 \
    --dir_40sweep ../../occupancy/nuscenes/v3/output/gt_pose/scene-0061 \
    --out_dir paper_out

# Fig 3: Pose 比較（GT pose vs KISS-SLAM）
python3 fig3_pose_comparison.py \
    --dir_gt   ../../occupancy/nuscenes/v3/output/gt_pose/scene-0061 \
    --dir_kiss ../../occupancy/nuscenes/v3/output/kiss_slam/scene-0061 \
    --out_dir paper_out
```

## 輸出位置

預設輸出到 `paper_out/`，把圖片複製到 `paper/` 目錄後 LaTeX 即可引用。

## 指定特定幀

加上 `--token <token_id>` 可指定要用哪一幀，否則預設取中間幀。
