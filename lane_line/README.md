# Lane Line Ground Truth Pipeline

獨立的車道線 ground truth 生成流程，與 occupancy 分開存放。
車道線不是占用（occupancy）資訊，而是路面幾何標記，因此作為獨立資料集使用。

---

## 資料流程

```
原始偵測 (per-frame JSON)
    ↓
多幀疊合 + KISS-ICP 對齊 (accumulate_lanes.py)
    ↓
側向分群 + 多項式擬合 (fit_lanes.py)
    ↓
fitted_lanes.json  ← 最終輸出（世界座標，per-scene）
```

---

## 輸出格式：`fitted_lanes.json`

**存放位置**：`lane_line/output/fitted/<scene_name>/fitted_lanes.json`

**座標系**：LiDAR / ego 慣例
- `x` = forward（車輛前進方向）
- `y` = lateral（左負右正）
- `z` = up（高度）
- 座標系為 **世界座標（KISS-ICP global frame）**，需搭配 `pose_dict.pkl` 轉到 ego frame

**格式**：
```json
[
  {
    "lane_id": 0,
    "points": [
      [x0, y0, z0],
      [x1, y1, z1],
      ...
    ],
    "n_pts": 500
  },
  ...
]
```

**Training 使用方式**：
```python
import json, numpy as np, pickle

with open('fitted_lanes.json') as f:
    lanes = json.load(f)

with open('pose_dict.pkl', 'rb') as f:
    pose_dict = pickle.load(f)

# 轉到某幀的 ego frame
T_inv = np.linalg.inv(pose_dict[frame_id]['matrix'])  # world → ego

for lane in lanes:
    pts_world = np.array(lane['points'])               # Nx3 world frame
    ones = np.ones((len(pts_world), 1))
    pts_ego = (T_inv @ np.hstack([pts_world, ones]).T).T[:, :3]  # Nx3 ego frame
```

---

## 腳本說明

### `accumulate_lanes.py`
多幀疊合原始車道線偵測結果，輸出 PLY 點雲與 BEV 圖。
支援 0413（per-lane with track_id）和 0429（flat xyz）兩種格式。

```bash
python3 lane_line/accumulate_lanes.py \
    --scene highway_sunny_day_2026-04-20-12-58-47
```

### `fit_lanes.py`
對疊合後的點雲進行車道線分群與多項式擬合，輸出 `fitted_lanes.json`。
**詳細演算法說明見 [FITTING.md](FITTING.md)。**

```bash
python3 lane_line/fit_lanes.py \
    --scene highway_sunny_day_2026-04-20-12-58-47 \
    --min_pts 20 \
    --degree 2 \
    --range_x -5 180 \
    --range_y -15 15
```

| 參數 | 說明 |
|------|------|
| `--min_pts` | 分群最少點數（預設 20） |
| `--degree` | 多項式次數（2=直線近似，適合高速公路） |
| `--max_x` | 截斷世界座標 x（m），避免 KISS-ICP 漂移影響 |

### `inject_fitted_lanes.py`
（選用）將 fitted lanes 注入 occupancy npz，產生含 label 18 的副本。
如果只用 JSON 做 training 則不需要執行。

---

## 目前處理的場景

| 場景 | 格式 | 車道數 | 說明 |
|------|------|--------|------|
| `highway_sunny_day_2026-04-20-12-58-47` | 0429 | 5 | 高速公路，實線×2 + 虛線×3 |

---

## 資料位置

```
lane_line/
  accumulate_lanes.py     # 疊合腳本
  fit_lanes.py            # 擬合腳本
  inject_fitted_lanes.py  # 注入 occupancy（選用）
  output/
    fitted/
      <scene_name>/
        fitted_lanes.json       # 最終車道線輸出
        lane_overlay.png        # 疊合比對圖
        lane_fitted_clean.png   # 乾淨擬合線圖
        lane_fit_comparison.png # 原始 vs 擬合對比圖
```
