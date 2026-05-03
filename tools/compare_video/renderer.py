#!/usr/bin/env python3
"""
renderer.py — Universal occupancy comparison video renderer.

Layout (1920x1080 fixed canvas):
  Left area  (MAIN_W):
    Top    : camera strip (CAM_H) — 1 or 6 cameras
    Bottom : N occupancy panels side-by-side (OCC_H)
  Right area (BEV_COL_W):
    BEV top-down view (square, centred)

Camera source is supplied as a callable:
    get_cameras(frame_id) -> dict or None
        Single cam  : {'FRONT': '/path/to/img.jpg'}
        Six cams    : {'FRONT': ..., 'FRONT_LEFT': ..., ...}
        Returns None / empty dict if no camera for this frame.

Supported camera key sets:
    1-cam  : any single key
    6-cam  : FRONT_LEFT, FRONT, FRONT_RIGHT, BACK_LEFT, BACK, BACK_RIGHT
"""

import os
import numpy as np
import cv2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Occ3D colour table (BGR)
# ---------------------------------------------------------------------------
OCC3D_COLORS_BGR = np.array([
    [  0,   0,   0],   # 0:  noise
    [144, 128, 112],   # 1:  barrier
    [ 43, 191, 235],   # 2:  bicycle
    [ 33,  91, 255],   # 3:  bus
    [ 60,  20, 220],   # 4:  car
    [ 33,  91, 255],   # 5:  construction_vehicle
    [  0,  61, 135],   # 6:  motorcycle
    [ 76,   0,  75],   # 7:  pedestrian
    [144, 128, 112],   # 8:  traffic_cone
    [ 33,  91, 255],   # 9:  trailer
    [ 33,  91, 255],   # 10: truck
    [248, 150, 215],   # 11: driveable_surface
    [ 70, 206, 247],   # 12: other_flat
    [ 70, 206, 247],   # 13: sidewalk
    [152, 251, 152],   # 14: terrain
    [ 70, 206, 247],   # 15: manmade
    [152, 251, 152],   # 16: vegetation
    [255, 255, 255],   # 17: free_space (not rendered)
    [  0,   0,   0],   # 18: lane line (black)
], dtype=np.uint8)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOXEL_SIZE  = 0.4
BG_COLOR    = (50, 50, 50)
VIEW_RANGE  = 35.0   # metres, fixed scale ±35m around ego

# Pre-compute ego car voxel xy (z computed at render time from z_offset)
_car_ex = np.arange(94, 106)
_car_ey = np.arange(97, 103)
_car_ez = np.arange(5, 9)
_eg = np.stack(np.meshgrid(_car_ex, _car_ey, _car_ez, indexing='ij'), axis=-1).reshape(-1, 3)
_EGO_PX   = (_eg[:, 0] - 100.0) * VOXEL_SIZE
_EGO_PY   = (_eg[:, 1] - 100.0) * VOXEL_SIZE
_EGO_EZ   = _eg[:, 2]   # raw z indices, z_offset applied at render time

_BLOCK_CACHE = {}   # block_size -> (offsets, bf)

CAM_6_KEYS = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT',
               'BACK_LEFT',  'BACK',  'BACK_RIGHT']
CAM_6_LABELS = ['Front Left', 'Front', 'Front Right',
                 'Back Left',  'Back',  'Back Right']


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _build_block_offsets(block, shaded=False):
    offsets, brightness = [], []
    for dy in range(block):
        face_b = 1.0 - 0.5 * (dy / max(block - 1, 1)) if shaded else 1.0
        for dx in range(block):
            is_edge = shaded and (dx == 0 or dx == block - 1 or dy == block - 1)
            offsets.append((dx, dy))
            brightness.append(face_b * 0.5 if is_edge else face_b)
    return (np.array(offsets, dtype=np.int32),
            np.array(brightness, dtype=np.float32))


def render_chase_cam(grid, elev_deg, occ_h, occ_w, voxel_style='flat', z_offset=-2.0):
    """3D chase-cam render of a (200,200,Z) occupancy grid."""
    occupied = (grid != 17) & (grid != 0) & (grid != 18)  # exclude lane lines from 3D view
    if not np.any(occupied):
        return np.full((occ_h, occ_w, 3), BG_COLOR, dtype=np.uint8)

    xs, ys, zs = np.where(occupied)
    px = (xs - 100.0) * VOXEL_SIZE
    py = (ys - 100.0) * VOXEL_SIZE
    pz = zs * VOXEL_SIZE + z_offset

    labels = grid[xs, ys, zs]
    colors = OCC3D_COLORS_BGR[labels].astype(np.float32)
    z_norm = (pz - pz.min()) / (pz.max() - pz.min() + 1e-6)
    if voxel_style == 'shaded':
        colors = np.clip(colors * (0.35 + 0.65 * z_norm)[:, np.newaxis], 0, 255).astype(np.uint8)
    else:  # flat
        colors = np.clip(colors * (0.6 + 0.4 * z_norm)[:, np.newaxis], 0, 255).astype(np.uint8)

    ego_pz = _EGO_EZ * VOXEL_SIZE + z_offset
    ez_norm = (ego_pz - ego_pz.min()) / (ego_pz.max() - ego_pz.min() + 1e-6)
    ego_colors = np.clip(
        np.array([50, 255, 0], dtype=np.float64) * (0.35 + 0.65 * ez_norm)[:, np.newaxis],
        0, 255).astype(np.uint8)

    px = np.concatenate([px, _EGO_PX])
    py = np.concatenate([py, _EGO_PY])
    pz = np.concatenate([pz, ego_pz])
    colors = np.concatenate([colors, ego_colors])

    elev = np.radians(elev_deg)
    se, ce = np.sin(elev), np.cos(elev)
    x_2d  = -py
    y_2d  =  px * se + pz * ce
    depth = -px * ce + pz * se

    scale = occ_w / (2.0 * VIEW_RANGE)
    block_size = max(4, int(np.ceil(VOXEL_SIZE * scale)))
    cache_key = (block_size, voxel_style)
    if cache_key not in _BLOCK_CACHE:
        _BLOCK_CACHE[cache_key] = _build_block_offsets(block_size, shaded=(voxel_style == 'shaded'))
    offsets, bf = _BLOCK_CACHE[cache_key]

    ego_scr_x = occ_w // 2
    ego_scr_y = int(occ_h * 0.70)
    x_screen = (x_2d * scale + ego_scr_x).astype(np.int32)
    y_screen = (ego_scr_y - y_2d * scale).astype(np.int32)

    margin = block_size
    visible = ((x_screen >= -margin) & (x_screen < occ_w + margin) &
               (y_screen >= -margin) & (y_screen < occ_h + margin))
    x_screen, y_screen = x_screen[visible], y_screen[visible]
    colors, depth = colors[visible], depth[visible]

    order = np.argsort(depth)
    x_s, y_s, c_s = x_screen[order], y_screen[order], colors[order]

    n_v, n_o = len(x_s), len(offsets)
    x_all = np.repeat(x_s, n_o) + np.tile(offsets[:, 0], n_v)
    y_all = np.repeat(y_s, n_o) + np.tile(offsets[:, 1], n_v)
    c_all = np.clip(
        np.repeat(c_s, n_o, axis=0).astype(np.float32) * np.tile(bf, n_v)[:, np.newaxis],
        0, 255).astype(np.uint8)

    valid = (x_all >= 0) & (x_all < occ_w) & (y_all >= 0) & (y_all < occ_h)
    canvas = np.full((occ_h, occ_w, 3), BG_COLOR, dtype=np.uint8)
    canvas[y_all[valid], x_all[valid]] = c_all[valid]
    return canvas


# ---------------------------------------------------------------------------
# FSD-style 3D perspective view
# ---------------------------------------------------------------------------
FSD_BG_SKY     = np.array([12, 10,  8], dtype=np.float32)
FSD_BG_FAR     = np.array([32, 36, 40], dtype=np.float32)
FSD_BG_NEAR    = np.array([42, 46, 50], dtype=np.float32)
FSD_ROAD_COLOR = (30, 35, 40)
FSD_LANE_COLOR = (80, 220, 255)
FSD_BOX_COLOR  = (200, 210, 220)
FSD_EDGE_COLOR = (255, 255, 255)
FSD_EGO_COLOR      = (200, 150,   0)
FSD_EGO_EDGE_COLOR = (255, 220, 50)
FSD_EGO_BOX        = np.array([2.5, 0.0, 0.8, 4.5, 2.0, 1.6, 0.0])

_FSD_BOX_EDGES = [(0,1),(2,3),(4,5),(6,7),
                  (0,2),(1,3),(4,6),(5,7),
                  (0,4),(1,5),(2,6),(3,7)]
_FSD_BOX_BOTTOM = [0,2,3,1]


def _fsd_fill_bg(canvas):
    h, w = canvas.shape[:2]
    horizon_y = int(h * 0.22)
    ys = np.arange(h, dtype=np.float32)
    t_above = np.clip(ys / max(1, horizon_y), 0, 1)
    t_below = np.clip((ys - horizon_y) / max(1, h - horizon_y), 0, 1)
    above = ys < horizon_y
    colors = np.where(above[:, None],
                      FSD_BG_SKY[None] * (1 - t_above[:, None]) + FSD_BG_FAR[None] * t_above[:, None],
                      FSD_BG_FAR[None] * (1 - t_below[:, None]) + FSD_BG_NEAR[None] * t_below[:, None]
                      ).astype(np.uint8)
    canvas[:] = colors[:, None, :]


def _fsd_make_proj(elev_deg, w, h, fov_deg=65.0, cam_height=8.0, cam_back=18.0):
    el = np.radians(elev_deg)
    se, ce = np.sin(el), np.cos(el)
    f = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    cx = w // 2
    cy = int(h * 0.78)

    def proj(pts):
        px = pts[:, 0] + cam_back
        py = pts[:, 1]
        pz = pts[:, 2] - cam_height
        x_cam = -py
        y_cam = -se * px - ce * pz
        z_cam =  ce * px - se * pz
        z_cam = np.maximum(z_cam, 0.5)
        sx = (f * x_cam / z_cam + cx).astype(np.int32)
        sy = (f * y_cam / z_cam + cy).astype(np.int32)
        return np.stack([sx, sy], axis=1), z_cam

    return proj


def _fsd_box_corners(box):
    cx, cy, cz, dx, dy, dz, h = box
    c, s = np.cos(h), np.sin(h)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    offsets = np.array([[ 1, 1, 1], [ 1, 1,-1], [ 1,-1, 1], [ 1,-1,-1],
                        [-1, 1, 1], [-1, 1,-1], [-1,-1, 1], [-1,-1,-1]],
                       dtype=np.float64) * np.array([dx/2, dy/2, dz/2])
    return (R @ offsets.T).T + np.array([cx, cy, cz])


def _fsd_draw_road(canvas, proj, view_range, back_range, road_width):
    hw = road_width / 2
    corners = np.array([[ view_range, -hw, 0],
                        [ view_range,  hw, 0],
                        [-back_range,  hw, 0],
                        [-back_range, -hw, 0]])
    pts_2d, _ = proj(corners)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts_2d.reshape(-1, 1, 2)], FSD_ROAD_COLOR)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)


def _fsd_draw_lane(canvas, proj, pts_ego, view_range, back_range, thickness=3):
    mask = ((pts_ego[:, 0] >= -back_range) & (pts_ego[:, 0] <= view_range) &
            (np.abs(pts_ego[:, 1]) <= view_range))
    pts_ego = pts_ego[mask]
    if len(pts_ego) < 2:
        return
    order = np.argsort(pts_ego[:, 0])
    pts_ego = pts_ego[order]
    scr, _ = proj(pts_ego)
    cv2.polylines(canvas, [scr.reshape(-1, 1, 2)], False, FSD_LANE_COLOR,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _fsd_draw_box(canvas, proj, box, fill_color, edge_color, thickness=2):
    corners = _fsd_box_corners(box)
    scr, _ = proj(corners)
    bottom = scr[_FSD_BOX_BOTTOM]
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [bottom.reshape(-1, 1, 2)], fill_color)
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    for i, j in _FSD_BOX_EDGES:
        cv2.line(canvas, tuple(scr[i]), tuple(scr[j]),
                 edge_color, thickness, cv2.LINE_AA)


def _filter_ego_self_detection(boxes):
    """Drop boxes whose centre lies inside ego footprint."""
    if boxes is None or not len(boxes):
        return boxes
    cx, cy = boxes[:, 0], boxes[:, 1]
    keep = ~((cx > -1.0) & (cx < 5.5) & (np.abs(cy) < 1.5))
    return boxes[keep]


def render_fsd_view(panel_w, panel_h, lane_pts_ego_list, boxes_ego,
                    elev_deg=14.0, view_range=80.0, back_range=5.0, road_width=28.0):
    """Tesla FSD-style 3D perspective panel.

    lane_pts_ego_list : list of Nx3 arrays already in ego frame
    boxes_ego         : Nx>=7 array of [x,y,z,dx,dy,dz,heading,...] in ego frame, or None
                        Ego self-detection is filtered out automatically.
    """
    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    _fsd_fill_bg(canvas)
    proj = _fsd_make_proj(elev_deg, panel_w, panel_h)
    _fsd_draw_road(canvas, proj, view_range, back_range, road_width)
    for lane in lane_pts_ego_list:
        _fsd_draw_lane(canvas, proj, lane, view_range, back_range)
    boxes_ego = _filter_ego_self_detection(boxes_ego)
    if boxes_ego is not None and len(boxes_ego):
        for b in boxes_ego:
            _fsd_draw_box(canvas, proj, b[:7], FSD_BOX_COLOR, FSD_EDGE_COLOR)
    # Always draw ego (own-vehicle visualization, drawn last so it's on top)
    _fsd_draw_box(canvas, proj, FSD_EGO_BOX, FSD_EGO_COLOR, FSD_EGO_EDGE_COLOR)
    return canvas


def render_bev(grid, size):
    """Top-down BEV of a (200,200,Z) grid, returns square (size x size) image."""
    # Pass 1: collapse all non-free, non-lane labels (vehicles, road, etc.)
    bev_map = np.ones((grid.shape[0], grid.shape[1]), dtype=np.uint8) * 17
    for z in range(grid.shape[2]):
        sl = grid[:, :, z]
        mask = (sl != 17) & (sl != 18)
        bev_map[mask] = sl[mask]
    # Pass 2: lane lines only where underlying label is road (11) or free (17)
    for z in range(grid.shape[2]):
        sl = grid[:, :, z]
        lane_mask = (sl == 18) & np.isin(bev_map, [17, 11])
        bev_map[lane_mask] = 18

    bev_rgb = OCC3D_COLORS_BGR[bev_map]
    bev_display = np.transpose(bev_rgb, (1, 0, 2))
    bev_display = np.flipud(bev_display)
    bev_display = np.rot90(bev_display, k=1)
    img = cv2.resize(bev_display, (size, size), interpolation=cv2.INTER_NEAREST)

    cx, cy = size // 2, size // 2
    cv2.rectangle(img, (cx - 6, cy - 10), (cx + 6, cy + 10), (0, 255, 0), -1)
    cv2.putText(img, 'BEV', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


def _load_grid(npz_path, grid_shape=(200, 200, 20)):
    if npz_path and os.path.exists(npz_path):
        return np.load(npz_path)['semantics']
    return np.ones(grid_shape, dtype=np.uint8) * 17


def _render_cam_strip_1(cam_paths, strip_w, strip_h):
    """Single camera: scale to fill strip_w, preserve aspect ratio, pad black top/bottom."""
    strip = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
    path = next(iter(cam_paths.values()), None)
    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            h0, w0 = img.shape[:2]
            fit_h = min(strip_h, int(strip_w * h0 / w0))
            fit_w = int(fit_h * w0 / h0)
            img = cv2.resize(img, (fit_w, fit_h))
            y_off = (strip_h - fit_h) // 2
            strip[y_off:y_off + fit_h, :fit_w] = img
    cv2.putText(strip, 'Front Camera', (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return strip


def _render_cam_strip_6(cam_paths, strip_w, strip_h):
    """Six cameras in 2 rows of 3: [FL, F, FR] / [BL, B, BR]."""
    row_h = strip_h // 2
    cam_w = strip_w // 3
    strip = np.full((strip_h, strip_w, 3), 30, dtype=np.uint8)

    for i, (key, label) in enumerate(zip(CAM_6_KEYS, CAM_6_LABELS)):
        row, col = divmod(i, 3)
        y0, x0 = row * row_h, col * cam_w
        path = cam_paths.get(key)
        if path and os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (cam_w, row_h))
                cv2.putText(img, label, (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                strip[y0:y0 + row_h, x0:x0 + cam_w] = img

    cv2.line(strip, (0, row_h), (strip_w, row_h), (60, 60, 60), 1)
    for col in range(1, 3):
        cv2.line(strip, (col * cam_w, 0), (col * cam_w, strip_h), (60, 60, 60), 1)
    return strip


def _render_cam_strip(cam_paths, strip_w, strip_h):
    """Auto-detect 1-cam or 6-cam layout."""
    if not cam_paths:
        return np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
    keys = set(cam_paths.keys())
    # Only use 6-cam layout if more than one camera key is present
    if len(keys) > 1 and keys & set(CAM_6_KEYS):
        return _render_cam_strip_6(cam_paths, strip_w, strip_h)
    return _render_cam_strip_1(cam_paths, strip_w, strip_h)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def _overlay_lanes(grid, lane_pts_world, T_inv, z_offset):
    """
    Transform world-frame lane points to ego frame, voxelize, and overlay
    label 18 (lane line) on a copy of grid — only overwriting road (11) or free (17).
    Returns modified grid copy.
    """
    GT_BOUNDS = np.array([-40.0, -40.0, -3.0])
    GT_VOXEL  = 0.4
    GT_GRID   = grid.shape
    LBL_ROAD, LBL_FREE, LBL_LANE = 11, 17, 18

    occ = grid.copy()
    for lane_w in lane_pts_world:
        ones = np.ones((len(lane_w), 1))
        pts_ego = (T_inv @ np.hstack([lane_w, ones]).T).T[:, :3]
        ix = ((pts_ego[:, 0] - GT_BOUNDS[0]) / GT_VOXEL).astype(np.int32)
        iy = ((pts_ego[:, 1] - GT_BOUNDS[1]) / GT_VOXEL).astype(np.int32)
        iz = ((pts_ego[:, 2] - GT_BOUNDS[2]) / GT_VOXEL).astype(np.int32)
        valid = ((ix >= 0) & (ix < GT_GRID[0]) &
                 (iy >= 0) & (iy < GT_GRID[1]) &
                 (iz >= 0) & (iz < GT_GRID[2]))
        ix, iy, iz = ix[valid], iy[valid], iz[valid]
        for dz in range(3):
            iz_c = np.clip(iz + dz, 0, GT_GRID[2] - 1)
            can  = np.isin(occ[ix, iy, iz_c], [LBL_FREE, LBL_ROAD])
            occ[ix[can], iy[can], iz_c[can]] = LBL_LANE
    return occ


def render_comparison_video(
    frame_ids,
    occ_dirs,
    panel_labels,
    bev_dir,
    get_cameras,
    out_path,
    fps=10,
    elev=28,
    total_w=1920,
    total_h=1080,
    cam_h=500,
    bev_col_w=500,
    bar_h=36,
    grid_shape=(200, 200, 20),
    voxel_style='flat',
    z_offset=-2.0,
    lane_pts_world=None,   # list of Nx3 arrays in world frame (fitted lanes)
    pose_dict=None,        # {frame_id: {'matrix': 4x4}} for lane projection
):
    """
    Render a comparison video.

    Parameters
    ----------
    frame_ids     : list of str   — frame identifiers (used as folder names)
    occ_dirs      : list of str   — one output dir per panel (scene level)
    panel_labels  : list of str   — label shown above each panel
    bev_dir       : str           — dir for BEV panel (scene level)
    get_cameras   : callable      — get_cameras(frame_id) -> dict[key, path]
                                    keys: any single key OR CAM_6_KEYS subset
    out_path      : str           — output .mp4 path
    fps           : int
    elev          : float         — elevation angle for 3D chase-cam (degrees)
    total_w/h     : int           — fixed canvas size
    cam_h         : int           — camera strip height
    bev_col_w     : int           — BEV column width (BEV will be square,
                                    as large as min(bev_col_w, total_h))
    bar_h         : int           — label bar height above each occ panel
    grid_shape    : tuple         — fallback empty grid shape
    """
    bev_size    = min(bev_col_w, total_h)
    main_w      = total_w - bev_col_w
    occ_h       = total_h - cam_h
    n_panels    = len(occ_dirs)
    panel_w     = main_w // n_panels
    content_h   = occ_h - bar_h

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (total_w, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {out_path}")

    print(f"Canvas {total_w}x{total_h} | main={main_w} bev_col={bev_col_w} "
          f"bev_size={bev_size} cam_h={cam_h} occ={panel_w}x{occ_h} x{n_panels}")

    for frame_id in tqdm(frame_ids, desc='Rendering'):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # --- Camera strip ---
        cam_paths = get_cameras(frame_id) or {}
        cam_strip = _render_cam_strip(cam_paths, main_w, cam_h)
        canvas[:cam_h, :main_w] = cam_strip

        # --- Occupancy panels ---
        for col_i, (d, lbl) in enumerate(zip(occ_dirs, panel_labels)):
            x0  = col_i * panel_w
            npz = os.path.join(d, frame_id, 'labels.npz')
            has_data = os.path.exists(npz)
            img = render_chase_cam(_load_grid(npz, grid_shape), elev, content_h, panel_w, voxel_style, z_offset=z_offset)
            if not has_data:
                cv2.putText(img, 'No Data', (panel_w // 2 - 60, content_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
            bar = np.zeros((bar_h, panel_w, 3), dtype=np.uint8)
            cv2.putText(bar, lbl, (8, bar_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            canvas[cam_h:, x0:x0 + panel_w] = np.vstack([bar, img])

        # Dividers between panels
        for col_i in range(1, n_panels):
            x = col_i * panel_w
            cv2.line(canvas, (x, cam_h), (x, total_h), (80, 80, 80), 2)

        # Frame label
        cv2.putText(canvas, str(frame_id),
                    (main_w - 130, total_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # --- BEV panel ---
        bev_npz  = os.path.join(bev_dir, frame_id, 'labels.npz')
        bev_grid = _load_grid(bev_npz, grid_shape)
        # Overlay fitted lane lines on-the-fly (no baking into npz)
        if lane_pts_world and pose_dict and frame_id in pose_dict:
            T_inv    = np.linalg.inv(pose_dict[frame_id]['matrix'])
            bev_grid = _overlay_lanes(bev_grid, lane_pts_world, T_inv, z_offset)
        bev_img  = render_bev(bev_grid, bev_size)
        if not os.path.exists(bev_npz):
            cv2.putText(bev_img, 'No Data', (bev_size // 2 - 60, bev_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
        bev_col = np.zeros((total_h, bev_col_w, 3), dtype=np.uint8)
        y_off   = (total_h - bev_size) // 2
        bev_col[y_off:y_off + bev_size, :bev_size] = bev_img
        cv2.line(bev_col, (0, 0), (0, total_h), (80, 80, 80), 2)
        canvas[:, main_w:] = bev_col

        writer.write(canvas)

    writer.release()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Three-view layout: top = camera strip, bottom = [Main 3D | 360 BEV | FSD]
# ---------------------------------------------------------------------------

def render_three_view_video(
    frame_ids,
    main_occ_dir,
    occ_360_dir,
    get_cameras,
    out_path,
    box_by_frame=None,
    score_thresh=0.4,
    lane_pts_world=None,
    pose_dict=None,
    fps=10,
    elev=28,
    fsd_elev=14.0,
    total_w=1920,
    total_h=1080,
    cam_h=500,
    bar_h=36,
    grid_shape=(200, 200, 20),
    voxel_style='shaded',
    z_offset=-2.0,
):
    """
    Three-view layout (top: 6-cam strip; bottom: 3 panels)
      Panel 1 (Main)   : chase-cam render of main_occ_dir (front-only semantics)
      Panel 2 (360)    : chase-cam render of occ_360_dir  (360° semantics)
      Panel 3 (FSD)    : Tesla FSD-style perspective with lanes + boxes + ego
    """
    panel_w   = total_w // 3
    occ_h     = total_h - cam_h
    content_h = occ_h - bar_h
    cam_strip_w = panel_w * 3

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (total_w, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {out_path}")

    print(f"Three-view canvas {total_w}x{total_h} | cam_h={cam_h} "
          f"panels=3 x {panel_w}x{content_h} | voxel_style={voxel_style}")

    labels = ['Main', '360', 'FSD']

    for frame_id in tqdm(frame_ids, desc='Rendering'):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # ── Camera strip ──
        cam_paths = get_cameras(frame_id) or {}
        cam_strip = _render_cam_strip(cam_paths, cam_strip_w, cam_h)
        canvas[:cam_h, :cam_strip_w] = cam_strip

        # ── Load both occupancy grids ──
        main_grid = _load_grid(os.path.join(main_occ_dir, frame_id, 'labels.npz'),
                               grid_shape)
        grid_360  = _load_grid(os.path.join(occ_360_dir,  frame_id, 'labels.npz'),
                               grid_shape)

        # ── Pose & lane→ego (only needed for FSD panel) ──
        lane_pts_ego = []
        if lane_pts_world and pose_dict and frame_id in pose_dict:
            T_inv = np.linalg.inv(pose_dict[frame_id]['matrix'])
            for lane_w in lane_pts_world:
                ones = np.ones((len(lane_w), 1))
                lane_pts_ego.append((T_inv @ np.hstack([lane_w, ones]).T).T[:, :3])

        # ── Boxes for FSD ──
        boxes_fsd = None
        if box_by_frame:
            try:
                key = str(int(frame_id))
            except ValueError:
                key = frame_id
            if key in box_by_frame:
                b = box_by_frame[key]
                m = b['score'] >= score_thresh
                boxes_fsd = b['boxes_lidar'][m]

        # ── Render the three panels ──
        img_main = render_chase_cam(main_grid, elev, content_h, panel_w,
                                    voxel_style, z_offset=z_offset)
        img_360  = render_chase_cam(grid_360,  elev, content_h, panel_w,
                                    voxel_style, z_offset=z_offset)
        img_fsd  = render_fsd_view(panel_w, content_h,
                                   lane_pts_ego, boxes_fsd, elev_deg=fsd_elev)

        for i, (lbl, img) in enumerate(zip(labels, [img_main, img_360, img_fsd])):
            x0  = i * panel_w
            bar = np.zeros((bar_h, panel_w, 3), dtype=np.uint8)
            cv2.putText(bar, lbl, (8, bar_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            canvas[cam_h:, x0:x0 + panel_w] = np.vstack([bar, img])

        for i in range(1, 3):
            x = i * panel_w
            cv2.line(canvas, (x, cam_h), (x, total_h), (80, 80, 80), 2)

        cv2.putText(canvas, str(frame_id),
                    (total_w - 130, total_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        writer.write(canvas)

    writer.release()
    print(f"Saved: {out_path}")
