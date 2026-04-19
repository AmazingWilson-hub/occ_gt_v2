# Paper Changelog

## 2026-04-19

---

### Abstract

**新增內容**（加在原摘要結尾）：

> The same accumulated semantic point cloud is also rasterized into a $200\times200\times16$ 3D occupancy grid; with ground-truth ego poses the pipeline achieves 72.27\% mIoU on nuScenes, and 58.20\% with KISS-SLAM estimated poses, demonstrating practical occupancy label generation without ground-truth localization.

---

### Introduction

**動機段落新增一句**（加在台灣資料集描述之後）：

> Furthermore, real-world deployment datasets often lack ground-truth ego poses, making multi-frame point cloud accumulation---a key step for dense occupancy label generation---dependent on estimated localization.

**Contribution 列表新增第四點**：

> \item We extend the pipeline to generate dense 3D semantic occupancy labels via adaptive multi-frame LiDAR accumulation, dynamic object volume filling, and a pose estimation module that supports datasets without ground-truth localization, achieving 72.27\% mIoU (GT pose) and 58.20\% mIoU (KISS-SLAM) on nuScenes.

---

### Methodology

#### Section 3.4 BEV Map Generation

- 標題從 `BEV Map Generation and Optional Occupancy Rasterization` 改為 `BEV Map Generation`
- 加入 `\label{sec:bev}`
- 刪除原本這句：
  > The same raster can also be converted into a simplified occupancy map by thresholding the number of points in each cell. However, this paper does not evaluate occupancy-specific metrics, so the quantitative analysis focuses on the semantic BEV map and projected semantic point cloud.

---

#### Section 3.5 3D Semantic Occupancy Label Generation（全新新增）

完整內容如下：

```latex
\subsection{3D Semantic Occupancy Label Generation}
\label{sec:occ}
In parallel with BEV map generation, we also produce a dense 3D occupancy grid directly from the accumulated multi-frame LiDAR point cloud and its semantic labels. The grid is defined in the ego-vehicle frame with voxel resolution $r_{\mathrm{occ}} = 0.4$\,m, spanning $[-40, 40] \times [-40, 40] \times [-1, 5.4]$\,m along the ego $X$, $Y$, and $Z$ axes, yielding a $200 \times 200 \times 16$ dense voxel volume. Each LiDAR point $\mathbf{p} = (x, y, z)$ is mapped to a voxel index $(i, j, k)$ by
\begin{equation}
  i = \left\lfloor \frac{x - x_{\min}}{r_{\mathrm{occ}}} \right\rfloor, \quad
  j = \left\lfloor \frac{y - y_{\min}}{r_{\mathrm{occ}}} \right\rfloor, \quad
  k = \left\lfloor \frac{z - z_{\min}}{r_{\mathrm{occ}}} \right\rfloor,
\end{equation}
where $(x_{\min}, y_{\min}, z_{\min}) = (-40, -40, -1)$\,m are the grid lower bounds. Points that fall outside the grid range are discarded.

\textbf{Label mapping.} The 32 nuScenes \texttt{lidarseg} classes are remapped to the 17 Occ3D semantic categories: ten dynamic classes (car, truck, bus, motorcycle, bicycle, pedestrian, traffic cone, barrier, construction vehicle, trailer) and six static classes (driveable surface, sidewalk, other flat, terrain, manmade, vegetation), plus an \emph{others} class for remaining foreground. Voxels that receive no point projection are assigned label~17 (\emph{free}).

\textbf{Adaptive multi-frame accumulation.} Spatial coverage is improved by stacking $\pm N$ surrounding LiDAR sweeps, but static and dynamic points require fundamentally different accumulation strategies.

For \emph{static points} (road, sidewalk, vegetation, manmade structures, etc.), we transform each historical sweep into the current ego frame using the full pose chain: LiDAR $\to$ ego $\to$ global $\to$ ego$_{\text{current}}$. This rigid transformation is valid because static elements do not move between frames, so accumulating them improves spatial density without introducing semantic inconsistency. A longer window ($N = 20$ sweeps) is used to maximize coverage of large ground and structure regions.

For \emph{dynamic objects} (vehicles, pedestrians, etc.), ego-motion compensation alone is insufficient because the objects themselves move independently. Instead, we align each instance across frames using its 3D bounding box pose: points belonging to the same tracked instance in a past or future frame are first transformed into the box-local coordinate frame of that frame, then re-expressed in the box-local frame of the current frame, and finally projected back into the current ego frame. This box-based alignment ensures that accumulated dynamic points remain geometrically consistent with the object's current position and orientation. A shorter window ($N = 5$ sweeps) is used for dynamic categories to limit ghosting from rapidly moving objects.

\textbf{Dynamic object volume filling.} For each 3D bounding box in the current frame, voxel centers that lie inside the box are identified using an axis-aligned bounding box test followed by a per-voxel point-in-box check in box-local coordinates. Empty voxels (label~17) within the box are then assigned the corresponding semantic class. This step converts hollow LiDAR surface shells into solid object volumes, improving coverage on the interior of large vehicles.

\textbf{Ego pose estimation for datasets without ground-truth localization.} Multi-frame accumulation requires accurate ego poses to transform past and future sweeps into the current frame. On nuScenes this is straightforward because ground-truth ego poses are provided. However, the custom Taiwan dataset does not supply ground-truth localization, which is a common constraint in real-world deployment. To address this, we evaluate several pose estimation backends as drop-in replacements: (i)~\emph{IMU dead reckoning}, which integrates CAN-bus acceleration and orientation but accumulates drift rapidly; (ii)~\emph{LiDAR ICP}, which estimates relative poses by sequential point-to-plane ICP between consecutive scans; (iii)~\emph{GPS+IMU EKF fusion}, which corrects IMU drift using GPS absolute position via an Extended Kalman Filter; and (iv)~\emph{full fusion}, which combines ICP-derived relative poses with EKF-corrected GPS positions to achieve sub-metre translation error. On nuScenes, KISS-SLAM is used as a representative learned odometry backend to simulate the no-GT-pose condition. For the custom Taiwan dataset, which provides neither GT ego poses nor loop-closure constraints, we adopt KISS-SLAM combined with GPS position fusion as the practical pose backend, as this combination provides the best trade-off between drift suppression and deployment simplicity. The choice of pose backend directly affects occupancy label quality because drift causes misalignment between accumulated sweeps, producing ghost voxels in static regions and blurred object boundaries.

The resulting $200 \times 200 \times 16$ occupancy tensor is evaluated against the nuScenes ground truth using the \texttt{mask\_lidar} field to restrict comparison to LiDAR-visible voxels.
```

---

### Experiments

#### Section 4.7 3D Occupancy Label Evaluation（全新新增）

完整內容如下：

```latex
\subsection{3D Occupancy Label Evaluation}
Table~\ref{tab:occupancy} reports the mIoU of the generated 3D occupancy labels against the nuScenes ground truth on the validation split (17 semantic classes, \texttt{mask\_lidar} evaluation protocol). We compare three pipeline configurations: (i)~a 10-sweep baseline with ground-truth (GT) ego poses, (ii)~a 40-sweep configuration with GT poses as an algorithmic upper bound, and (iii)~a 40-sweep configuration using estimated KISS-SLAM poses, representing the practical deployment setting without access to GT localization.

\begin{table}[t]
  \caption{3D occupancy label quality on the nuScenes validation split (17 semantic classes, \texttt{mask\_lidar} evaluation).}
  \label{tab:occupancy}
  \centering
  \begin{tabular}{llcc}
    \toprule
    Configuration & Pose Backend & Sweeps & mIoU (\%) \\
    \midrule
    Baseline & GT pose      & 10 & 48.82 \\
    Ours     & GT pose      & 40 & 72.27 \\
    Ours     & KISS-SLAM    & 40 & 58.20 \\
    \bottomrule
  \end{tabular}
\end{table}

The GT-pose upper bound of 72.27\% demonstrates that adaptive multi-frame stacking and dynamic box volume filling can produce high-quality occupancy labels when accurate ego poses are available. In the practical KISS-SLAM setting, mIoU decreases to 58.20\%, indicating that pose drift is the primary remaining source of error. The gap between the 10-sweep baseline (48.82\%) and the 40-sweep variant (72.27\% with GT pose) confirms that longer temporal accumulation substantially improves coverage and label completeness in the 3D occupancy representation.
```

---

### References

- 嘗試加入 Occ3D (NeurIPS 2023) reference，後依使用者要求移除（venue 不確定）
