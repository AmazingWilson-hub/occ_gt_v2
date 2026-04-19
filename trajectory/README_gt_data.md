# Trajectory GT Data

## 資料位置
`data/trajectory/gt/`

## 資料來源
GAI_v4 格式，3 個場景（citystreet，sunny day/night）

## 檔案列表
```
citystreet_sunny_day_2025-10-13-17-04-09_GAI_v4_gt_only.json
citystreet_sunny_night_2025-10-13-19-08-40_GAI_v4_gt_only.json
citystreet_sunny_night_2025-10-13-19-09-42_GAI_v4_gt_only.json
```

## JSON 格式
```json
{
  "frame_000000": {
    "gt":   [[x, y], [x, y], ...],
    "mask": [true, true, ...]
  }
}
```
- 每個檔案共 89 frames（frame_000000 ~ frame_000088）
- 座標以自車為原點：**x = 前進距離（m）**，**y = 側向偏移（m）**
- 共 6 個 waypoint，間距約 5~6 m，涵蓋前方約 33 m
- mask 標記各 waypoint 是否有效

## 用途
作為軌跡預測的 ground truth，與模型輸出比對評估。

## TODO
- [ ] 撰寫評估腳本，讀取此 GT 與預測結果做比對（ADE / FDE 等 metrics）
