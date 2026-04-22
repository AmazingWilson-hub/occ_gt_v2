# main_v2.tex 修改狀態

## 已完成的修改

### Sug 1｜Problem statement：缺乏 ego-pose → impractical
**修改位置**：Introduction 第二段

- 把「problem becomes more severe」改為明確的 impractical 陳述
- 加入「estimated poses introduce drift that propagates into every accumulated sweep」
- 引用 OCC-VO，說明 occupancy quality ↔ pose accuracy 的雙向耦合關係
- 新增 `\cite{occvo}` 進 bibliography

### Sug 2｜Problem statement：calibration 錯誤放大迭代誤差
**修改位置**：Introduction 第一段末尾

- 在 calibration 相關文獻引用後，新增一句：
  「minor calibration drift...accumulates across multi-frame stacking and progressively amplifies semantic label errors in the voxelized output」

**修改位置**：Contribution bullet 2

- 把「instead of discarding them」改為
  「providing a fault-tolerant buffer against extrinsic calibration drift that would otherwise be amplified through multi-frame accumulation」
- 明確將 Potential Obstacle 定位為 calibration drift 的修正機制

### Sug 5｜Experiments 加入實驗設置描述
**修改位置**：Section III 開頭

- 新增 `\textbf{Setup.}` 段落
- 說明：硬體平台（VLS-128 LiDAR）、實作環境（Python/PyTorch）
- 明確定義三個評估協定（SAL 6-class / nuScenes 16-class / Occ3D mask\_lidar 17-class）
- 解釋 `mask\_lidar` 的作用（限制在 LiDAR-observable voxels，避免 empty-space 膨脹 mIoU）
- 說明 pose backend 選擇（GT vs KISS-SLAM）

### Sug 6｜Result 開新 subsection
**修改位置**：Table 之後的結果段落

- 加入 `\textbf{Results.}` inline heading

### Sug 7｜Result and discussion 進行討論
**修改位置**：Results 段落之後

- 新增 `\textbf{Discussion.}` 段落，內容包含：
  - 13.5-point gap（GT vs KISS-SLAM）的成因分析（pose drift，非 semantic 問題）
  - 連結 OCC-VO 的 occupancy ↔ pose coupling 論點
  - 台灣資料集 qualitative 結果的限制說明
  - 明確說明 external occupancy baseline 缺失的原因（無直接可比的 17-class mask\_lidar 方法），並說明這是 future work

---

## 尚未完成（需要新實驗或新圖）

### Sug 3 / 8 / 10｜Occupancy 需要外部 baseline
**問題**：Table 1 的 Dense occupancy representation 三行全是 "Ours"，無外部方法對比

**需要做什麼**：
找一個在 nuScenes 上有 17-class occupancy mIoU（mask\_lidar 協定）數字的方法，選項如下：

| 選項 | 難度 | 說明 |
|---|---|---|
| 純 LiDAR single-frame voxelization（無語意投影） | 低 | 直接跑，作為 lower bound；量化語意投影的貢獻 |
| Occ3D GT generation pipeline（官方） | 中 | Occ3D 論文附帶的 GT 生成流程，可當 upper bound 參照 |
| SAL 輸出 → voxelization | 中 | 把 SAL 語意點雲丟進你的 voxelization 流程，直接比較語意品質對 occupancy mIoU 的影響 |
| OCC-VO | 不適用 | OCC-VO 的 occupancy 指標是 Accuracy/Precision/Completion，不是 mIoU，無法直接比較 |

**最小工作量方案**：跑「SAL 語意點雲 → 同一個 voxelization pipeline」，得到 SAL 的 occupancy mIoU，加入 Table 1。這樣三行 "Ours" 就有 SAL baseline 可以比較。

---

### Sug 9｜Fig. 3 視覺化需要加對比
**問題**：目前三張子圖（camera inputs / 3D occupancy / BEV）純展示，無對比

**需要做什麼**：
把其中一張替換或新增並排比較，建議仿照 OCC-VO Table III 的格式：

```
| Scene | Ours (40 sweeps) | 10-sweep baseline | GT |
```

需要生成的新圖：
- 10-sweep occupancy 的 BEV 或 3D view（同一個 scene）
- GT occupancy 的對應 BEV（從 nuScenes Occ3D gts/ 讀取）

相關腳本已存在：`tools/paper_figures/fig2_sweep_comparison.py`，可以直接擴充或參考。

---

### 其他注意事項

- **OCC-VO 的 `\cite{occvo}` 已加入 bibliography**，但這是 arXiv preprint（cs.RO），
  投稿時需確認期刊是否接受 arXiv 引用，或查是否已有正式發表版本

- **Sweep 數描述與程式碼不一致**：
  Paper 說「靜態 N=20，動態 N=5」，但 v4/generate.py 實際是「driveable_surface 40, 其他靜態 10」
  目前 main_v2.tex 的 Proposed Framework 段落用的是模糊描述（"longer/shorter window"），
  尚未修正為精確數字，暫時安全，但若審稿人詳問需要補充
