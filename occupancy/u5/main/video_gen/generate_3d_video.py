#!/usr/bin/env python3
"""
Generate combined 3D occupancy + 6-camera video (chase-cam style).
Layout:
  Row 1: Front Left | Front | Front Right
  Row 2: Back Left  | Back  | Back Right
  Bottom: 3D Occupancy chase-cam with 3D ego car body
"""
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse

OCC3D_COLORS_BGR = np.array([
    [0, 0, 0],             # 0:  noise
    [144, 128, 112],       # 1:  barrier
    [235, 191,  43],       # 2:  bicycle
    [ 33,  91, 255],       # 3:  bus
    [ 60,  20, 220],       # 4:  car
    [ 33,  91, 255],       # 5:  construction_vehicle
    [  0,  61, 135],       # 6:  motorcycle
    [ 75,   0,  76],       # 7:  pedestrian
    [144, 128, 112],       # 8:  traffic_cone
    [ 33,  91, 255],       # 9:  trailer
    [ 33,  91, 255],       # 10: truck
    [248, 150, 215],       # 11: driveable_surface
    [ 70, 206, 247],       # 12: other_flat
    [ 70, 206, 247],       # 13: sidewalk
    [152, 251, 152],       # 14: terrain
    [ 70, 206, 247],       # 15: manmade
    [152, 251, 152],       # 16: vegetation
    [255, 255, 255],       # 17: free_space (skip)
], dtype=np.uint8)

VOXEL_SIZE = 0.4
BG_COLOR = (25, 25, 25)

# Camera layout: two rows
CAM_ROW_TOP = [
    (['cam_front_left',  'port_2_camera'], 'Front Left'),
    (['cam_front',       'port_8_camera'], 'Front'),
    (['cam_front_right', 'port_5_camera'], 'Front Right'),
]
CAM_ROW_BOT = [
    (['cam_back_left',   'port_3_camera'], 'Back Left'),
    (['cam_back',        'port_7_camera'], 'Back'),
    (['cam_back_right',  'port_6_camera'], 'Back Right'),
]


def _build_block_offsets(block):
    """Pixel offsets + brightness for shaded voxel blocks."""
    offsets, brightness = [], []
    for dy in range(block):
        face_b = 1.25 - 0.55 * (dy / max(block - 1, 1))
        for dx in range(block):
            is_edge = (dx == 0 or dx == block - 1 or dy == 0 or dy == block - 1)
            offsets.append((dx, dy))
            brightness.append(face_b * 0.45 if is_edge else face_b)
    return (np.array(offsets, dtype=np.int32),
            np.array(brightness, dtype=np.float32))


def _find_cam_image(scene_dir, dir_candidates, frame):
    """Find camera image with fallback naming."""
    for cand in dir_candidates:
        p = os.path.join(scene_dir, cand, f"{frame}.jpg")
        if os.path.exists(p):
            return p
    return None


def render_cam_strip(scene_dir, frame, strip_w, row_h):
    """Render 6 cameras in 2 rows of 3."""
    cam_w = strip_w // 3
    strip = np.full((row_h * 2, strip_w, 3), 30, dtype=np.uint8)

    for row_idx, cam_row in enumerate([CAM_ROW_TOP, CAM_ROW_BOT]):
        y_off = row_idx * row_h
        for col_idx, (dirs, label) in enumerate(cam_row):
            img_path = _find_cam_image(scene_dir, dirs, frame)
            if img_path is not None:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (cam_w, row_h))
                    cv2.putText(img, label, (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                    strip[y_off:y_off + row_h,
                          col_idx * cam_w:(col_idx + 1) * cam_w] = img

    # Separator lines
    cv2.line(strip, (0, row_h - 1), (strip_w, row_h - 1), (60, 60, 60), 1)
    cv2.line(strip, (0, row_h * 2 - 1), (strip_w, row_h * 2 - 1), (80, 80, 80), 1)
    return strip


def render_chase_cam(grid, elev_deg, occ_h, occ_w):
    """Chase-cam 3D render. Ego car is rendered as green voxels in the same pipeline."""
    occupied = (grid != 17) & (grid != 0)
    if not np.any(occupied):
        return np.full((occ_h, occ_w, 3), BG_COLOR, dtype=np.uint8)

    xs, ys, zs = np.where(occupied)

    # --- Physical coords for scene voxels ---
    px = (xs - 100.0) * VOXEL_SIZE
    py = (ys - 100.0) * VOXEL_SIZE
    pz = zs * VOXEL_SIZE + (-2.0)

    # Scene colors with height shading
    labels = grid[xs, ys, zs]
    colors = OCC3D_COLORS_BGR[labels].astype(np.float64)
    z_norm = (pz - pz.min()) / (pz.max() - pz.min() + 1e-6)
    base_bright = 0.6 + 0.4 * z_norm
    colors = np.clip(colors * base_bright[:, np.newaxis], 0, 255).astype(np.uint8)

    # --- Inject ego car as green voxels (same rendering pipeline = true 3D) ---
    # Car: ~4.8m long, ~2.0m wide, ~1.6m tall, centered at grid (100,100)
    # Grid z=5 is physical z=0 (ground), z=9 is z=1.6m (roof)
    car_ex = np.arange(94, 106)     # 12 voxels = 4.8m length
    car_ey = np.arange(97, 103)     # 6 voxels = 2.4m width (slightly wide for visibility)
    car_ez = np.arange(5, 9)        # 4 voxels = 1.6m height
    ego_grid = np.stack(np.meshgrid(car_ex, car_ey, car_ez, indexing='ij'), axis=-1).reshape(-1, 3)
    ego_px = (ego_grid[:, 0] - 100.0) * VOXEL_SIZE
    ego_py = (ego_grid[:, 1] - 100.0) * VOXEL_SIZE
    ego_pz = ego_grid[:, 2] * VOXEL_SIZE + (-2.0)

    # Green with height shading (same formula)
    ego_z_norm = (ego_pz - ego_pz.min()) / (ego_pz.max() - ego_pz.min() + 1e-6)
    ego_bright = 0.6 + 0.4 * ego_z_norm
    ego_base = np.array([50, 255, 0], dtype=np.float64)  # BGR green
    ego_colors = np.clip(ego_base * ego_bright[:, np.newaxis], 0, 255).astype(np.uint8)

    # Combine scene + ego voxels
    px = np.concatenate([px, ego_px])
    py = np.concatenate([py, ego_py])
    pz = np.concatenate([pz, ego_pz])
    colors = np.concatenate([colors, ego_colors])

    # --- Chase cam projection ---
    elev = np.radians(elev_deg)
    se, ce = np.sin(elev), np.cos(elev)

    x_2d = -py
    y_2d = px * se + pz * ce
    depth = -px * ce + pz * se

    ego_x2d, ego_y2d = 0.0, 0.0
    x_left  = ego_x2d - x_2d.min()
    x_right = x_2d.max() - ego_x2d
    y_above = y_2d.max() - ego_y2d
    y_below = ego_y2d - y_2d.min()

    ego_frac_x, ego_frac_y = 0.50, 0.65
    margin = 0.04

    scale = min(
        occ_w * (ego_frac_x - margin) / max(x_left, 0.1),
        occ_w * (1.0 - ego_frac_x - margin) / max(x_right, 0.1),
        occ_h * (ego_frac_y - margin) / max(y_above, 0.1),
        occ_h * (1.0 - ego_frac_y - margin) / max(y_below, 0.1),
    )

    ego_scr_x = int(occ_w * ego_frac_x)
    ego_scr_y = int(occ_h * ego_frac_y)

    x_screen = ((x_2d - ego_x2d) * scale + ego_scr_x).astype(np.int32)
    y_screen = (ego_scr_y - (y_2d - ego_y2d) * scale).astype(np.int32)

    order = np.argsort(depth)
    x_sorted = x_screen[order]
    y_sorted = y_screen[order]
    colors_sorted = colors[order]

    block = max(4, int(np.ceil(VOXEL_SIZE * scale * 1.15)))
    offsets, bf = _build_block_offsets(block)
    n_vox = len(x_sorted)
    n_off = len(offsets)

    x_all = np.repeat(x_sorted, n_off) + np.tile(offsets[:, 0], n_vox)
    y_all = np.repeat(y_sorted, n_off) + np.tile(offsets[:, 1], n_vox)
    c_base = np.repeat(colors_sorted, n_off, axis=0).astype(np.float64)
    bf_tiled = np.tile(bf, n_vox)
    c_all = np.clip(c_base * bf_tiled[:, np.newaxis], 0, 255).astype(np.uint8)

    valid = (x_all >= 0) & (x_all < occ_w) & (y_all >= 0) & (y_all < occ_h)

    canvas = np.full((occ_h, occ_w, 3), BG_COLOR, dtype=np.uint8)
    canvas[y_all[valid], x_all[valid]] = c_all[valid]

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generate 3D + 6cam combined video")
    parser.add_argument('--scene', default='test_2026-03-23-10-42-37')
    parser.add_argument('--data_root',
                        default='/home/t113c52027/t113c52027/occ_gt_v2/data/u5')
    parser.add_argument('--bev_root',
                        default='/home/t113c52027/t113c52027/occ_gt_v2/cvpr_format_occ_gen_u5/output')
    parser.add_argument('--out_dir',
                        default='/home/t113c52027/t113c52027/occ_gt_v2/cvpr_format_occ_gen_u5/video_gen')
    parser.add_argument('--mode', default='heuristic')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--elev', type=float, default=25)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.scene}_{args.mode}_3d.mp4')

    scene_dir = os.path.join(args.data_root, args.scene)
    pred_dir = os.path.join(args.bev_root, args.scene, args.mode)
    frames = sorted([d for d in os.listdir(pred_dir)
                     if os.path.exists(os.path.join(pred_dir, d, 'labels.npz'))])
    if not frames:
        print(f"ERROR: No labels.npz found in {pred_dir}")
        return
    print(f"Found {len(frames)} frames → {out_path}")

    # Layout: cameras keep native 3:2 aspect ratio
    total_w = 1920
    cam_w = total_w // 3                          # 640
    cam_row_h = int(cam_w * 1280 / 1920)          # 427 (exact 3:2)
    cam_strip_h = cam_row_h * 2                   # 854
    occ_h = 720                                   # 3D occupancy area
    total_h = cam_strip_h + occ_h                 # 1574
    print(f"Canvas: {total_w}x{total_h} (cam {cam_w}x{cam_row_h} x2 rows + occ {occ_h})")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, args.fps, (total_w, total_h))

    for i, frame in enumerate(tqdm(frames, desc="Rendering 3D + 6cam")):
        # Top: 6 cameras (2 rows × 3) — native aspect ratio
        cam_strip = render_cam_strip(scene_dir, frame, total_w, cam_row_h)

        # Bottom: 3D occupancy (directly below cameras, no gap)
        grid = np.load(os.path.join(pred_dir, frame, 'labels.npz'))['semantics']
        occ_img = render_chase_cam(grid, elev_deg=args.elev,
                                   occ_h=occ_h, occ_w=total_w)

        # Text overlays directly on the 3D area (no black gap)
        cv2.putText(occ_img, f"3D Occupancy ({args.mode})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(occ_img, f"Frame: {frame}",
                    (10, occ_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        full_frame = np.vstack([cam_strip, occ_img])
        out.write(full_frame)

    out.release()
    print(f"Done! Video saved to {out_path}")


if __name__ == '__main__':
    main()
