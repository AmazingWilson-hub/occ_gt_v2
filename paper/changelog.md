# Paper Changelog

## 2026-04-19

### Abstract
- 結尾新增一句：說明 pipeline 也能產生 200×200×16 的 3D occupancy grid，並報告 mIoU 結果（GT pose: 72.27%，KISS-SLAM: 58.20%）

### Introduction
- 動機段落新增一句：說明真實部署資料集（如台灣自訂資料集）常缺乏 GT ego pose，使多幀點雲累積依賴估計定位
- Contribution 列表新增第四點：涵蓋 3D occupancy label generation、adaptive multi-frame 累積、dynamic object volume filling、pose estimation module，以及具體 mIoU 數字

### Methodology

#### Section 3.4 BEV Map Generation（原 3.4）
- 標題從 `BEV Map Generation and Optional Occupancy Rasterization` 改為 `BEV Map Generation`
- 加入 `\label{sec:bev}`
- 刪除原本那句 "the same raster can also be converted into a simplified occupancy map... this paper does not evaluate occupancy-specific metrics"

#### Section 3.5 3D Semantic Occupancy Label Generation（新增）
- Grid 規格：200×200×16，0.4m 體素，XY ±40m，Z -1~5.4m
- Voxelization 公式：LiDAR 點座標 → 體素索引 (i, j, k)
- Label mapping：lidarseg 32 類 → 17 語意類別，無點體素給 free（label 17）
- Adaptive multi-frame accumulation：
  - 靜態點：用完整 ego pose 變換鏈對齊，N=20
  - 動態物件：用 3D bounding box pose 跨幀對齊同一實例，N=5
- Dynamic object volume filling：AABB test + 逐體素點在框內測試，空閒體素填入對應語意類別
- Ego pose estimation：
  - 列舉四種後端（IMU dead reckoning、LiDAR ICP、GPS+IMU EKF、full fusion）
  - nuScenes 用 KISS-SLAM 模擬無 GT pose 情境
  - 台灣資料集採用 KISS-SLAM + GPS fusion

### Experiments

#### Section 4.7 3D Occupancy Label Evaluation（新增）
- 新增 Table 4：比較三個配置的 mIoU
  - Baseline / GT pose / 10 sweeps → 48.82%
  - Ours / GT pose / 40 sweeps → 72.27%
  - Ours / KISS-SLAM / 40 sweeps → 58.20%
- 分析：GT pose 上界說明方法有效；KISS-SLAM 下降說明 pose drift 是主要誤差來源；sweep 數增加顯著提升覆蓋率

### References
- 嘗試加入 Occ3D reference，後依使用者要求移除
