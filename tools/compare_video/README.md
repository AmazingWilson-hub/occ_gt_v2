# compare_video — Universal Occupancy Comparison Video Tool

並排比較多種 occupancy 生成方法的影片工具，支援任意欄數、多種相機來源，固定 1920×1080 畫布。

## 版面配置

```
┌──────────────────────────────────┬───────────┐
│  相機畫面 (cam_h)                 │           │
├──────────┬──────────┬────────────┤  BEV 俯視  │
│ Panel 0  │ Panel 1  │  Panel N   │  (正方形)  │
│ (3D occ) │ (3D occ) │  (3D occ) │           │
└──────────┴──────────┴────────────┴───────────┘
         ← main_w →              ← bev_col_w →
```

- 畫布大小固定 `total_w × total_h`（預設 1920×1080）
- BEV 為正方形，`size = min(bev_col_w, total_h)`，垂直置中
- 相機支援 1 鏡頭或 6 鏡頭（自動偵測）
- 單鏡頭：填滿相機區寬度，保持比例，左右留黑邊
- 六鏡頭：2 排 3 格（FL/F/FR 上排，BL/B/BR 下排）

---

## 快速使用

### ELAN（單鏡頭）

```bash
python3 tools/compare_video/run.py \
    --dirs \
        cvpr_format_occ_gen_elan/output/citystreet.../raw \
        cvpr_format_occ_gen_elan/output/citystreet.../heuristic \
        cvpr_format_occ_gen_elan/output/citystreet.../seg \
    --labels "Raw" "Heuristic" "Semantic" \
    --bev_dir cvpr_format_occ_gen_elan/output/citystreet.../seg \
    --cam_dir data/elan/citystreet.../image \
    --grid_z   21
```

### G6（單鏡頭，自動偵測）

```bash
python3 tools/compare_video/run.py \
    --dirs \
        cvpr_format_occ_gen_g6/output/citystreet.../raw \
        cvpr_format_occ_gen_g6/output/citystreet.../heuristic \
        cvpr_format_occ_gen_g6/output/citystreet.../seg \
    --labels "Raw" "Heuristic" "Semantic" \
    --bev_dir cvpr_format_occ_gen_g6/output/citystreet.../seg \
    --cam_dir data/g6/citystreet.../image \
    --grid_z   21
```

### U5（6 鏡頭，目錄結構）

```bash
python3 tools/compare_video/run.py \
    --dirs \
        cvpr_format_occ_gen_u5/output/test_.../raw \
        cvpr_format_occ_gen_u5/output/test_.../heuristic \
        cvpr_format_occ_gen_u5test/output/test_.../semantic \
    --labels "Raw" "Heuristic" "Semantic" \
    --bev_dir cvpr_format_occ_gen_u5test/output/test_.../semantic \
    --cam_dir  data/u5/test_... \
    --grid_z   20
```

### NuScenes（6 鏡頭，API）

```bash
python3 tools/compare_video/run.py \
    --dirs \
        cvpr_format_occ_gen_v3/output/gt_pose/scene-0061 \
        cvpr_format_occ_gen_v4/output/kiss_slam_all10/kiss_slam/scene-0061 \
        cvpr_format_occ_gen_v4/output/kiss_slam/scene-0061 \
    --labels "GT pose" "KISS-SLAM all=10" "KISS-SLAM road=40, others=10" \
    --bev_dir cvpr_format_occ_gen_v4/output/kiss_slam/scene-0061 \
    --cam_mode nuscenes \
    --scene    scene-0061 \
    --grid_z   16
```

---

## 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--dirs` | 各欄的 occupancy 輸出目錄（scene level） | 必填 |
| `--labels` | 各欄標題，數量需與 `--dirs` 相同 | 必填 |
| `--bev_dir` | BEV 使用的 occupancy 目錄 | 必填 |
| `--out` | 輸出 .mp4 路徑（不指定則自動生成） | 自動 |
| `--cam_mode` | `auto` / `none` / `single` / `6cam_dir` / `nuscenes` | `auto` |
| `--cam_dir` | 相機圖片目錄（auto / single / 6cam_dir 模式） | - |
| `--cam_ext` | 相機圖片副檔名 | `.jpg` |
| `--scene` | NuScenes scene 名稱 | `scene-0061` |
| `--dataroot` | NuScenes 資料根目錄 | `/data2/.../nuscenes_occ` |
| `--fps` | 影片幀率 | `10` |
| `--elev` | 3D 視角仰角（度） | `28` |
| `--total_w/h` | 畫布尺寸 | `1920 / 1080` |
| `--cam_h` | 相機區高度（px） | `500` |
| `--bev_col_w` | BEV 欄位寬度（px） | `500` |
| `--grid_z` | Occupancy grid Z 層數（NuScenes=16, U5/ELAN/G6=20/21） | `20` |
| `--voxel_style` | `flat`=均勻亮度（預設）/ `shaded`=邊緣暗化 | `flat` |

---

## 相機來源（`--cam_mode`）

### `auto`（預設）
自動偵測：
- `--cam_dir` 未指定 → `none`
- `--cam_dir` 內直接有圖片 → `single`
- `--cam_dir` 內有 U5 子目錄（`port_*_camera` 等） → `6cam_dir`

### `single`
單一前視相機，圖片以 `{frame_id}.jpg` 命名存放在 `--cam_dir` 目錄下。
```
cam_dir/
  000000.jpg
  000001.jpg
  ...
```

### `6cam_dir`
U5 格式，六個相機各自放在子目錄下：
```
scene_dir/
  port_2_camera/000000.jpg   → FRONT_LEFT
  port_8_camera/000000.jpg   → FRONT
  port_5_camera/000000.jpg   → FRONT_RIGHT
  port_3_camera/000000.jpg   → BACK_LEFT
  port_7_camera/000000.jpg   → BACK
  port_6_camera/000000.jpg   → BACK_RIGHT
```

### `nuscenes`
透過 NuScenes DevKit API 讀取，frame_id 為 sample token，幀序依時間順序排列。

### `none`
不顯示相機。

---

## Voxel 渲染樣式（`--voxel_style`）

| 樣式 | 說明 |
|------|------|
| `flat` | 均勻亮度，最低 60%，無邊緣格線感（預設） |
| `shaded` | 邊緣暗化，有 3D 格子立體感，最低 35% |

---

## 程式化使用

```python
from tools.compare_video.renderer import render_comparison_video

def my_get_cameras(frame_id):
    return {'FRONT': f'/path/to/images/{frame_id}.jpg'}

render_comparison_video(
    frame_ids    = ['000000', '000001', ...],
    occ_dirs     = [dir_a, dir_b, dir_c],
    panel_labels = ['Method A', 'Method B', 'Method C'],
    bev_dir      = dir_c,
    get_cameras  = my_get_cameras,
    out_path     = 'output.mp4',
    grid_shape   = (200, 200, 20),
    voxel_style  = 'flat',
)
```
