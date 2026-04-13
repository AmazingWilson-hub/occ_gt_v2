# G6 Data Notes

## 已知問題

### citystreet_sunny_day_2026-03-02-14-33-32
- `result_depth_filtered/000040.pcd` — 檔案大小為 0 bytes，上傳時失敗
- 該幀語意點雲缺失，generate.py 會自動跳過（該幀 seg 輸出為全 free）
- 需要重新上傳該檔案後重跑 semantic 模式

## 場景說明

| 場景 | raw | heuristic | semantic | 備註 |
|------|-----|-----------|----------|------|
| 2026-02-03-15-17-34 | ✓ | ✓ | ✓ | |
| 2026-02-03-16-51-01 | ✓ | ✓ | ✓ | |
| 2026-02-03-17-00-10 | ✓ | ✓ | — | 無語意點雲資料 |
| 2026-02-03-17-03-09 | ✓ | ✓ | ✓ | |
| 2026-03-02-14-19-39 | ✓ | ✓ | ✓ | 語意為 result_depth_filtered（非 v2） |
| 2026-03-02-14-28-54 | ✓ | ✓ | ✓ | 語意為 result_depth_filtered（非 v2） |
| 2026-03-02-14-33-32 | ✓ | ✓ | ✓* | 語意為 result_depth_filtered（非 v2），000040.pcd 缺失 |
| 2026-03-02-14-13-43 | ✓ | ✓ | ✓ | 語意為 result_depth_filtered（非 v2），89 幀 |
| 2026-03-02-14-13-58 | ✓ | ✓ | ✓ | 語意為 result_depth_filtered（非 v2），89 幀 |
| 2026-03-02-14-14-33 | ✓ | ✓ | — | 語意只有 65/90 幀，不完整，跳過 semantic |
| 2026-03-02-14-16-53 | ✓ | ✓ | ✓ | 語意為 result_depth_filtered（非 v2），88 幀 |
| 2026-03-02-14-34-40 | ✓ | ✓ | ✓ | 語意為 result_depth_filtered（非 v2），90 幀 |
| 2026-03-02-14-35-14 | ✓ | ✓ | ✓ | 語意為 result_depth_filtered（非 v2），89 幀 |
| 2026-03-02-14-36-02 | ✓ | ✓ | ✓ | 語意為 result_depth_filtered（非 v2），89 幀 |
| 2026-03-02-15-06-09 | ✓ | ✓ | — | 語意只有 32/89 幀，不完整，跳過 semantic |
| 2026-03-02-15-06-36 | ✓ | ✓ | ✓ | 語意為 result_depth_filtered（非 v2），90 幀 |
| 2026-03-02-15-14-03 | ✓ | ✓ | ✓ | 語意為 result_depth_filtered（非 v2），89 幀 |
| 2026-03-09-10-47-58 | ✓ | ✓ | ✓ | 語意為 colored_360_pcd_filter，43 幀（VLS=42） |
| 2026-03-09-15-54-19 (rainy) | ✓ | ✓ | ✓ | 語意為 colored_360_pcd_filter，42 幀（VLS=41） |
| 2026-03-09-15-29-09 countryside rainy | ✓ | ✓ | ✓ | 語意為 colored_360_pcd_filter，43 幀（VLS=42） |
