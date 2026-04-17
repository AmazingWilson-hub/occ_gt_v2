# NuScenes Occupancy 生成流程說明文件

這份文件說明如何從 NuScenes 數據集生成 3D Occupancy Ground Truth (佔據網格真值)。

## 📂 目錄結構 (`occupancy_pipeline/`)

```
occupancy_pipeline/
├── run_pipeline.sh          # 一鍵自動化腳本 (生成 + 轉換)
├── data_converter.py        # 核心邏輯：生成稀疏 Occupancy (多幀堆疊)
├── convert_to_npz.py        # 格式轉換：.bin -> .npz (CVPR Challenge 格式)
├── visualize_occupancy.py   # 視覺化工具 (3D 點雲 & BEV 圖片)
└── utils/                   # 輔助函式庫 (幾何計算、對齊)
```

---

## 🚀 1. 自動化流程 (`run_pipeline.sh`)

**用途**：最簡單的全流程執行方式。它會自動依序執行「生成」與「格式轉換」。

**使用方式**：
```bash
bash run_pipeline.sh
```

**執行內容**：
1.  呼叫 `data_converter.py` 在 `output_bin/` 生成原始 `.pcd.bin` 檔案。
2.  呼叫 `convert_to_npz.py` 將其轉換為 `.npz` 格式 (並存於 `output_bin/` 或搬移)。

**設定**：
若需修改路徑或版本，請編輯腳本開頭的變數：
```bash
DATAROOT="../data/nuscenes_occ/" 
VERSION="v1.0-trainval"  # 或是 v1.0-mini 用於測試
```

---

## 🛠️ 2. 各步驟腳本詳解

### A. 生成原始資料：`data_converter.py`

**用途**：讀取 NuScenes 原始 LiDAR 與標註，執行多幀堆疊 (Multi-frame Stacking)，並移除動態物體殘影，建立靜態背景場景。

**輸入資料 (Input)**：
*   NuScenes 數據集 (v1.0-trainval 或 v1.0-mini)
*   **必要條件**：該目錄下必須包含 `lidarseg.json` (我們已手動修復過此問題)。

**輸出資料 (Output)**：
*   `.pcd.bin` 檔案：原始 Float16 陣列 `[N, 5]` -> `(x, y, z, intensity, label)`。

**手動執行指令**：
```bash
python3 data_converter.py \
  --dataroot ../data/nuscenes_occ/ \
  --save_path output_bin/ \
  --version v1.0-mini
```

**核心邏輯**：
*   **靜態堆疊**：累積過去與未來的非移動物體點雲。
*   **動態物體處理**：利用 Bounding Box 將移動物體嚴格對齊到當前時間戳的位置。

---

### B. 格式轉換：`convert_to_npz.py`

**用途**：將原始點雲轉換為 CVPR 2023 Challenge 標準的 **Semantic Occupancy Grid** 格式 (.npz)。

**執行動作**：
1.  **範圍過濾**：保留範圍內的點 `X/Y: [-60, 60]`, `Z: [-5, 11]`。
2.  **體素化 (Voxelization)**：將浮點數坐標量化為 `0.2m` 的整數索引。
3.  **儲存**：僅儲存稀疏索引 (Indices) 與對應標籤 (Semantics)。

**輸出資料 (Output)**：
*   `.npz` 壓縮檔，包含：
    *   `indices`: `[N, 3]` (int) - 體素網格坐標。
    *   `semantics`: `[N]` (int) - 類別標籤 (0-31)。

**手動執行指令**：
```bash
# 處理單一檔案
python3 convert_to_npz.py data/file.pcd.bin --out output.npz

# 批次處理整個目錄 (多行程平行加速)
python3 convert_to_npz.py output_bin/ --num_workers 16
```

---

### C. 視覺化驗證：`visualize_occupancy.py`

**用途**：驗證生成的資料是否正確。支援兩種格式。

**使用方式**：

1.  **查看 .npz (Challenge 格式)**
    *   *輸出彩色 PLY (可用 MeshLab/CloudCompare 開啟)*：
        ```bash
        python3 visualize_occupancy.py sample.npz --save_ply out.ply
        ```
    *   *輸出 BEV 鳥瞰圖 (圖片檔)*：
        ```bash
        python3 visualize_occupancy.py sample.npz --save_img out.png
        ```

2.  **查看 .pcd.bin (原始格式)**
    ```bash
    python3 visualize_occupancy.py sample.pcd.bin --save_ply out.ply --use_labels
    ```

**顏色配置**：
已內建 **NuScenes 官方配色表** (例如：道路=青色, 車輛=橘色, 植被=綠色)。

---

## 📊 數據規格 (Data Specifications)

| 屬性 | 數值 | 備註 |
| :--- | :--- | :--- |
| **Voxel Size** | `0.2m` | 體素大小 |
| **X/Y Range** | `[-60.0m, 60.0m]` | 前後左右範圍 |
| **Z Range** | `[-5.0m, 11.0m]` | 高度範圍 |
| **Classes** | 32 類 | 0=noise, 17=car, 24=driveable, 31=ego... |

## ⚠️ 常見問題與解決
1.  **"Missing lidarseg.json"**：這是因為 NuScenes 下載時未解壓 metadata。請至 archive 把 `lidarseg.json` 解壓到對應 version 目錄。
2.  **"道路少了一半 (Half road missing)"**：這是範圍設定錯誤。請確保使用更新後的 `convert_to_npz.py` (設定為 `[-60, 60]`)，不要用舊版 `[-40, 40]` 的設定。
