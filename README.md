# occ_gt_v2 — 自車軌跡估算 + 3D 語意佔用網格生成

從 LiDAR / IMU / GPS 等原始感測器數據，自動推算自車軌跡（Ego Pose），並生成符合 CVPR2023 Occ3D 格式的 3D 語意佔用網格（Semantic Occupancy Grid）。

---

## 目錄結構

```
occ_gt_v2/
├── README.md                          # 本文件
├── data/                              # 原始資料集
├── pose_estimation/                   # 自車軌跡推估（四種方法）
├── cvpr_format_occ_gen/               # Occupancy 生成 V1（官方 GT pose）
├── cvpr_format_occ_gen_v2/            # Occupancy 生成 V2（加入 Box Volume 填充）
├── cvpr_format_occ_gen_v3/            # Occupancy 生成 V3（結合 KISS-SLAM 姿態與 Box 填充）
├── cvpr_format_occ_gen_fusionego/     # Occupancy 生成（使用 Full Fusion 估計 pose）
├── elan_occ_gen/                      # ELAN 資料集 Occupancy 生成
├── occupancy_pipeline/                # 評估工具、除錯腳本、視覺化工具
├── presentation_exporter/             # 簡報用視覺化素材匯出
└── occupancy-for-nuscenes-main/       # 第三方參考程式碼（GitHub clone）
```

---

## 資料集路徑 (`data/`)

| 資料 | 路徑 |
|:---|:---|
| NuScenes (v1.0-mini) | `data/nuscenes_occ/` |
| NuScenes CAN Bus 數據 | `data/nuscenes_occ/can_bus/` |
| CVPR2023 Occ3D 官方 GT | `data/nuscenes_occ/gts/<scene>/<sample-token>/labels.npz` |

---

## 1. 自車軌跡推估 (`pose_estimation/`)

### 檔案

| 腳本 | 說明 |
|:---|:---|
| `estimate_ego_pose.py` | 核心程式。實作四種方法並對 GT 評估誤差 |
| `generate_occ_with_estimated_poses.py` | 早期整合版（已被 `cvpr_format_occ_gen_fusionego/` 取代，僅供參考） |

### 四種方法比較（NuScenes scene-0061）

| 方法 | 平移誤差 (m) | 旋轉誤差 (°) |
|:---|:---:|:---:|
| 1. IMU Dead Reckoning | 91.8 | 2.4 |
| 2. LiDAR ICP | 5.5 | 1.8 |
| 3. GPS + IMU EKF | 9.0 | 2.4 |
| **4. Full Fusion (ICP + IMU + GPS)** | **0.96** | **1.8** |

### Full Fusion 演算法流程

1. **IMU** 提供相鄰兩幀的旋轉初始猜測，讓 ICP 有準確的起點
2. **LiDAR ICP (Point-to-Plane)** 精確計算相對位移與旋轉
3. **座標系轉換**：`T_ego = T_cs @ T_icp @ T_cs_inv`（LiDAR → Ego frame）
4. **GPS EKF 修正位置漂移**：`K = P / (P + R_gps)`，只修正位置 (x, y, z)，旋轉完全信任 ICP

### 執行方式

```bash
cd /home/t113c52027/t113c52027/occ_gt_v2
python3 pose_estimation/estimate_ego_pose.py
```

---

## 2. Occupancy 生成（NuScenes）

### 通用輸出格式

```
輸出：<out_root>/<scene-name>/<sample-token>/labels.npz
格式：semantics 陣列 (200, 200, 16) uint8
範圍：X/Y = [-40m, 40m]，Z = [-1m, 5.4m]
Voxel：0.4m，標籤 0-16 = 語意類別，17 = 空（Free）
```

---

### `cvpr_format_occ_gen/` — V1 基準版（官方 GT ego_pose）

使用 **NuScenes 官方 ego_pose** 生成 Occupancy，包含動態物件 Bounding Box 對位處理。

**含有腳本：**

| 腳本 | 說明 |
|:---|:---|
| `generate.py` | 主程式：多幀堆疊 + 動態物件對位 + 體素化 |
| `evaluate.py` | 計算生成結果對官方 GT 的 mIoU |
| `diagnose.py` | 視覺化除錯工具 |

**執行：**

```bash
# 生成
python3 cvpr_format_occ_gen/generate.py \
  --dataroot data/nuscenes_occ \
  --version v1.0-mini \
  --scene_name scene-0061

# 評估 mIoU（對官方 GT）
python3 cvpr_format_occ_gen/evaluate.py \
  --pred_root cvpr_format_occ_gen/output \
  --gt_root data/nuscenes_occ/gts \
  --scene scene-0061
```

---

### `cvpr_format_occ_gen_v2/` — V2 改良版（Box Volume 填充）

V1 生成的動態物件體素只有 LiDAR 掃描到的表面點。V2 加入**「3D Bounding Box 體積填充」**：體素化後，把偵測到的動態物件 Box 內部所有空 (Free=17) 的 Voxel 全部填入對應語意標籤，讓動態物件更接近官方 GT 的「實心體積」標註。

**執行：**

```bash
python3 cvpr_format_occ_gen_v2/generate.py \
  --dataroot data/nuscenes_occ \
  --version v1.0-mini \
  --scene_name scene-0061
```

---

### `cvpr_format_occ_gen_v3/` — V3 終極進化版（極限 Sweeps + Box 填充）

V3 是目前實測效果**最強**的生成架構！將堆疊幀數拉到極限（**40 sweeps**），並且完整繼承了 V2 的動態物件「3D Box Volume 體積填充」能力。在 `scene-0061` 實測中，V3 搭配自主測計的 **KISS-SLAM Ego Pose** 能達到驚人的 **0.5820** mIoU！若直接使用官方完美的 GT Pose，更測出了 **0.7227** 的演算法理論上限，證明這套生成管線具有極高的 3D 還原能力。

**目前 V3 架構 (scene-0061) 測試成績：**

| 軌跡來源 (Pose Backend) | Sweeps | mIoU (16 classes) | 備註 |
|:---|:---:|:---:|:---|
| **KISS-SLAM (自主推估)** | 40 | **0.5820** | 完全無依賴官方軌跡的最佳實戰成績 |
| **GT Pose (官方軌跡)** | 40 | **0.7227** | 此管線的演算法理論上限 (Upper Bound) |

**含有腳本：**

| 腳本 | 說明 |
|:---|:---|
| `generate.py` | 主程式：可指定 `--backend` 推算軌跡 + 40 sweeps 生成 + 動態 Box 填充 |
| `visualize.py` | BEV 視覺化工具，生成 Occ3D CVPR 標準顏色的俯視圖 (`.png`) |

**執行生成與視覺化：**

```bash
# 自動使用 KISS-SLAM 測計軌跡 生成 40 sweeps 的超密實 Occupancy
python3 cvpr_format_occ_gen_v3/generate.py --backend kiss_slam --scene scene-0061

# 若要測試純理論上限 (使用 GT Pose)
python3 cvpr_format_occ_gen_v3/generate.py --backend gt_pose --scene scene-0061

# 分析結果並匯出 BEV 鳥瞰彩色圖
python3 cvpr_format_occ_gen_v3/visualize.py --pred_dir cvpr_format_occ_gen_v3/output/kiss_slam/scene-0061
```

---

### `cvpr_format_occ_gen_fusionego/` — Full Fusion Pose 版

**自動** 執行以下完整流程：
1. 呼叫 `pose_estimation/` 的 Full Fusion 演算法推算軌跡
2. 使用該軌跡生成 Occupancy（同 V1 的多幀堆疊與動態物件對位）
3. 與 GT ego_pose 版比較逐幀 mIoU（輸出 `miou_per_frame.npy`）

**執行：**

```bash
# 生成 Occupancy（使用 Full Fusion 軌跡）
python3 cvpr_format_occ_gen_fusionego/generate.py --scene scene-0061

# 評估 mIoU（對官方 Occ3D GT）
python3 occupancy_pipeline/evaluate_miou.py \
  --pred_root cvpr_format_occ_gen_fusionego/output \
  --gt_root data/nuscenes_occ/gts \
  --scene scene-0061
```

**scene-0061 評估結果（Full Fusion vs 官方 Occ3D GT）：**

| 類別 | IoU | 類別 | IoU |
|:---|:---:|:---|:---:|
| barrier | 0.7606 | driveable_surface | 0.4147 |
| bicycle | 0.6535 | other_flat | 0.4033 |
| car | 0.4669 | sidewalk | 0.2921 |
| pedestrian | 0.5712 | terrain | 0.3238 |
| truck | 0.6083 | manmade | 0.3252 |
| **Mean IoU (16 classes)** | **0.4882** | vegetation | 0.4462 |

---

## 3. ELAN 資料集 Occupancy 生成 (`elan_occ_gen/`)

針對 ELAN 自建資料集，**與 NuScenes 完全無關**，使用 ICP 自行推算 pose 後生成 Occupancy。

**資料格式：**
- 點雲：`VLS128_pcdnpy/*.pcd`（二進位 PCD，含 x/y/z/rgb）
- IMU：`imu/*.txt`（四元數姿態）
- GPS：`gps/*.txt`（緯度/經度/高度，部分幀缺失）
- 標籤：`label/*.json`（3D Bounding Box JSON）

| 腳本 | 說明 |
|:---|:---|
| `generate.py` | 主程式：ICP 算 pose + 生成 Occupancy |
| `export_all.py` | 批次匯出所有場景 |
| `export_semantic_pc.py` | 匯出語意點雲 |
| `gen_color_occ.py / v2 / v3` | 生成彩色 BEV 俯視圖（版本迭代） |
| `stack_semantic_pc.py` | 多幀語意點雲堆疊 |

---

## 4. 評估與工具 (`occupancy_pipeline/`)

| 腳本 / 檔案 | 說明 |
|:---|:---|
| **`evaluate_miou.py`** | **主要評估工具**：計算 `pred_root` vs `gt_root` 的 mIoU |
| `compare_occupancy.py` | 視覺化比較兩份 Occupancy（GT vs 生成結果） |
| `compare_stacking.py` | 比較不同幀數堆疊方式的差異 |
| `export_stacked_ply.py` | 匯出堆疊後的點雲為 PLY（供 CloudCompare 開啟） |
| `generate_gt_format.py` | 舊版生成腳本（已被 `cvpr_format_occ_gen/` 取代，僅參考） |
| `data_converter.py` | 最早期流程：生成稀疏 `.pcd.bin` 格式（已棄用） |
| `reproduce_algorithm.py` | 動態物件對齊的早期參考實作 |
| `visualize_occupancy.py` | 視覺化工具：支援 `.npz` 和 `.pcd.bin` 格式 |
| `visualize_bev.py` | BEV 俯視圖視覺化（通用版） |
| `visualize_bev_nuscenes.py` | BEV 俯視圖視覺化（NuScenes 官方配色） |
| `visualize_ply_raw.py` | PLY 點雲視覺化 |
| `debug_*.py` | 各種除錯工具（標籤、對齊、空間、Voxel 等） |

**evaluate_miou.py 使用方式：**

```bash
python3 occupancy_pipeline/evaluate_miou.py \
  --pred_root <你的 output 資料夾> \
  --gt_root data/nuscenes_occ/gts \
  --scene scene-0061
```

---

## 5. 簡報視覺化匯出 (`presentation_exporter/`)

生成簡報展示用的 Demo 場景截圖與渲染圖片。

```bash
bash presentation_exporter/run_exporter.sh
```

---

## 標籤定義（Occ3D 17 類 + 1 Empty）

| ID | 類別 | ID | 類別 |
|:---:|:---|:---:|:---|
| 0 | others (ignore) | 9 | trailer |
| 1 | barrier | 10 | truck |
| 2 | bicycle | 11 | driveable_surface |
| 3 | bus | 12 | other_flat |
| 4 | car | 13 | sidewalk |
| 5 | construction_vehicle | 14 | terrain |
| 6 | motorcycle | 15 | manmade |
| 7 | pedestrian | 16 | vegetation |
| 8 | traffic_cone | **17** | **free / empty** |

---

## 依賴套件

```bash
pip install nuscenes-devkit scipy numpy open3d tqdm pyquaternion
```
