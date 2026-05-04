# 車道線擬合方法

從原始 per-frame 光達車道線偵測，產出乾淨的解析曲線（`fitted_lanes.json`）。

實作：[lane_line/fit_lanes.py](fit_lanes.py)

---

## 流程總覽

```mermaid
flowchart TD
    A["原始偵測 JSON<br/>per-frame, ego frame, 雜亂"]
    B["累積點雲<br/>world frame, 五條混合"]
    C["分群結果<br/>每條獨立的點群"]
    D["擬合曲線<br/>每條 500 個解析取樣點"]
    E["fitted_lanes.json<br/>world frame, row-major"]

    A -->|pose_dict[fid] 變換| B
    B -->|"KDE on y → find_peaks → valleys"| C
    C -->|"polyfit(deg=2) + savgol"| D
    D -->|序列化| E
```

---

## 為什麼需要這套流程

原始輸入是每幀車道線偵測點，存在三個問題：

| 問題 | 說明 |
|------|------|
| 斷裂 | 高速公路虛線本來就斷，單幀只看到一截 |
| 稀疏 | 雷射打到車道線的點少 |
| 混雜 | 五條車道的點全混在一起，沒有 lane id |
| 雜訊 | 偵測器會抖 |

目標：把這團點轉成五條乾淨曲線，每條獨立、可數學表示。

---

## 1. 多幀疊合

KISS-ICP 對每幀算出 `T_world←ego` 的 4×4 變換矩陣。把每幀車道線投到世界座標再堆疊，就能組合出完整高速公路車道。

[fit_lanes.py:59-80](fit_lanes.py#L59-L80)

```python
def accumulate_lanes(scene):
    pose_dict = load_pose_dict(scene)        # KISS-ICP 結果
    pts_list = []
    for fid in sorted(pose_dict.keys()):
        pts = load_lane_0429(f'{fid}.json')  # ego frame Nx3
        T   = pose_dict[fid]['matrix']        # 4x4
        pts_list.append(transform_pts(T, pts))
    return np.vstack(pts_list)                # 全部堆成一個大陣列
```

座標細節：0429 JSON 的 `xyz` 是 column-major 且順序 `[lateral, forward, z]`，要重排成標準 `[x_fwd, y_lat, z]`。

### 踩過的坑：KISS-ICP 漂移

KISS-ICP 在遠端有累積誤差，到第 80 m 後 5 條車道的點會「擠成一團」。
解法：用低階多項式（`degree=2`）就能吸收漂移影響，不需要 `--max_x` 截斷。

---

## 2. 側向分群（KDE Clustering）— **流程關鍵**

### 為什麼標準分群法都不行

| 方法 | 失敗原因 |
|------|---------|
| K-Means | 需要事先知道 K |
| DBSCAN | ε 不好調；同條車道內部 x 跨度太大 |
| y histogram | bin 寬大會合併稀疏虛線；bin 寬小會切碎單條車道 |

### 解法：1D KDE on lateral axis + 山谷切割

把所有點投影到 y 軸，五條車道會變成五個密度峰值：

```
density
   ▲
   │      ╱╲       ╱╲                 ╱╲
   │     ╱  ╲     ╱  ╲    ╱╲   ╱╲   ╱  ╲
   │    ╱    ╲   ╱    ╲  ╱  ╲ ╱  ╲ ╱    ╲
   │___╱______╲_╱______╲╱____╳____╳______╲___▶ y
       lane1   lane2   lane3 lane4 lane5
                         │
                        山谷 = 切點
```

虛線的峰會比實線矮（點少），但只要峰存在就能找到。

### KDE 數學

[fit_lanes.py:101](fit_lanes.py#L101)

$$\hat{f}(y) = \frac{1}{nh}\sum_{i=1}^{n} K\!\left(\frac{y - y_i}{h}\right), \quad K(u) = \frac{1}{\sqrt{2\pi}}e^{-u^2/2}$$

把每個點當成小高斯凸起，全部加起來就是平滑的密度估計。**帶寬 h 是最關鍵超參數**：

| h | 效果 |
|---|------|
| 1.0 m | 所有峰糊在一起，5 條像 1 條 |
| **0.25 m** | **剛好分開鄰近車道（車道寬 ~3.5 m）** |
| 0.05 m | 每條內部都有一堆小峰，被切碎 |

`gaussian_kde` 的 `bw_method` 是相對於資料 std 的比例，所以實作上要除以 `ys.std()`：

```python
kde = gaussian_kde(ys, bw_method=0.25 / ys.std())
```

### 找峰

```python
y_grid  = np.linspace(y_min, y_max, 2000)
density = kde(y_grid)
peaks, _ = find_peaks(density,
                      distance=1.0 / dy,                    # 兩峰最少距 1 m
                      prominence=density.max() * 0.01)      # 突出度 ≥ 1% 最大密度
```

| 參數 | 作用 |
|------|------|
| `distance` | 避免一條車道因點分布不均產生兩個小峰被誤判 |
| `prominence` | 過濾雜訊小起伏；實線、虛線都過得了，雜訊過不了 |

### 找山谷（切點）

N 個峰之間有 N-1 個切點，每對相鄰峰之間找密度最小值：

```python
for i in range(len(peaks) - 1):
    lo, hi = peaks[i], peaks[i + 1]
    valley_idx = lo + np.argmin(density[lo:hi + 1])
    cut_ys.append(y_grid[valley_idx])
```

### 切割

```python
order = np.argsort(ys)
split_pos = np.searchsorted(ys[order], cut_ys)
groups = np.split(order, split_pos)
return [pts[g] for g in groups if len(g) >= min_pts]
```

---

## 3. 每條車道的多項式擬合

### 模型

把車道參數化為 `y = f(x)`、`z = g(x)`（前提：車道不會 U-turn，高速公路成立）。

### 程式

[fit_lanes.py:128-155](fit_lanes.py#L128-L155)

```python
def fit_lane(pts, degree=3, n_samples=300, smooth_window=21):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    coeff_y = np.polyfit(x, y, degree)
    coeff_z = np.polyfit(x, z, degree)
    x_clean = np.linspace(x.min(), x.max(), n_samples)
    y_clean = np.polyval(coeff_y, x_clean)
    z_clean = np.polyval(coeff_z, x_clean)
    if smooth_window:
        y_clean = savgol_filter(y_clean, smooth_window, 2)
        z_clean = savgol_filter(z_clean, smooth_window, 2)
    return np.stack([x_clean, y_clean, z_clean], axis=1)
```

### degree 選擇

| degree | 特性 | 高速公路適用？ |
|--------|------|---------------|
| 1 | 直線 | 直道 OK，彎道不行 |
| **2** | 拋物線 | **平緩彎道夠用，最 robust** |
| 3 | 立方 | 容易過擬合，遠端反翹 |
| 4 | 四次 | 嚴重過擬合，被 KISS-ICP 漂移帶歪 |

### Savgol 後處理

`savgol_filter` 是 Savitzky-Golay 濾波器：滑動窗口內套低階多項式擬合，用擬合中心值取代原值。

對 degree=2 的結果幾乎沒影響，但對 degree>=3 可以明顯抑制小波動。算 belt-and-suspenders。

---

## 4. 輸出格式

[fit_lanes.py:270-287](fit_lanes.py#L270-L287)

```json
[
  {"lane_id": 0, "points": [[x0,y0,z0], ...], "n_pts": 500},
  {"lane_id": 1, "points": [...], "n_pts": 500},
  ...
]
```

| 欄位 | 設計理由 |
|------|---------|
| `points` row-major | 直接 `np.array(lane['points'])` 拿到 Nx3，不用重組 |
| `lane_id` | 由 `cluster_by_lateral` 排序（按平均 y 從左到右） |
| 世界座標 | 下游用 `T_inv = inv(pose_dict[fid]['matrix'])` 轉回任一幀 ego frame |

---

## 設計哲學

**核心 trick**：把 3D 分群問題降維成 1D 問題（只看 y 軸密度）。

成立的先驗：
- 高速公路車道**平行**（相同 x 範圍）
- **lateral 分布唯一**（每條 y 中心固定）

換成市區交叉路口、匝道、轉彎多的場景這個假設會崩潰，要改用 polyline tracking、graph-based clustering 等通用方案。
但對 highway 場景，這是相對於問題複雜度最簡單也最 robust 的選擇。
