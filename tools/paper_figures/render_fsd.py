#!/usr/bin/env python3
"""
Tesla FSD-style 3D visualization:
  - Dark road surface
  - Bright lane lines
  - White vehicle boxes (from 3D box detections)
  - Perspective projection from behind/above

Usage:
    python3 tools/paper_figures/render_fsd.py \
        --scene highway_sunny_day_2026-04-20-12-58-47 \
        --out video_out/highway_fsd.mp4
"""

import os, sys, json, pickle, argparse
import numpy as np
import cv2

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

# ── Projection parameters ────────────────────────────────────────────────────
ELEV       = 14.0         # camera elevation angle (degrees) — low for FSD look
CAM_HEIGHT = 8.0          # camera height above ground (metres)
CAM_BACK   = 18.0         # camera offset behind ego (metres)
FOV_DEG    = 65.0         # horizontal field of view (degrees)
VIEW_RANGE = 80.0         # metres visible ahead
BACK_RANGE = 5.0          # metres visible behind ego (must be < CAM_BACK)
W, H       = 1280, 720

# ── Visual style (BGR) ───────────────────────────────────────────────────────
BG_COLOR       = ( 15,  15,  20)   # near-black background
ROAD_COLOR     = ( 30,  35,  40)   # dark road surface
LANE_COLOR     = ( 80, 220, 255)   # warm yellow-white lane lines (BGR)
EGO_COLOR      = (200, 150,   0)   # blue ego vehicle (BGR)
BOX_COLOR      = (200, 210, 220)   # light vehicle boxes
BOX_EDGE_COLOR = (255, 255, 255)
LANE_THICKNESS = 3
BOX_THICKNESS  = 2
ROAD_WIDTH     = 28.0              # metres, total road width rendered


# ── Geometry helpers ─────────────────────────────────────────────────────────

def make_proj(elev_deg, w, h, fov_deg=FOV_DEG, cam_height=CAM_HEIGHT, cam_back=CAM_BACK):
    """Perspective projection: (N,3) ego-pts → (N,2) screen xy + depth.

    Camera is placed cam_back metres behind ego at cam_height above ground,
    pointing forward and slightly down (elevation = elev_deg).
    """
    el  = np.radians(elev_deg)
    se, ce = np.sin(el), np.cos(el)
    # focal length from horizontal FOV
    f = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    cx = w // 2
    cy = int(h * 0.78)   # principal point — low horizon

    def proj(pts):
        # pts: Nx3 [x_fwd, y_lat, z_up]
        # Translate: camera is cam_back behind ego, cam_height up
        px = pts[:, 0] + cam_back   # forward distance from camera
        py = pts[:, 1]              # lateral (unchanged)
        pz = pts[:, 2] - cam_height # height relative to camera

        # Rotate by elevation (camera tilts down by elev_deg)
        # Camera X axis = lateral, Y axis = up-in-image, Z axis = into scene
        x_cam =  py
        y_cam = -(px * se + pz * ce)   # negative → down in image = positive sy
        z_cam =  px * ce - pz * se     # depth (positive = in front)

        # Clamp depth to avoid behind-camera artifacts
        z_cam = np.maximum(z_cam, 0.5)

        sx = ( f * x_cam / z_cam + cx).astype(np.int32)
        sy = (-f * y_cam / z_cam + cy).astype(np.int32)
        return np.stack([sx, sy], axis=1), z_cam

    return proj


def box_corners(box):
    """Return 8 corners (8x3) of a 3D box [x,y,z,dx,dy,dz,heading]."""
    cx, cy, cz, dx, dy, dz, h = box
    c, s = np.cos(h), np.sin(h)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    offsets = np.array([[ 1, 1, 1], [ 1, 1,-1], [ 1,-1, 1], [ 1,-1,-1],
                        [-1, 1, 1], [-1, 1,-1], [-1,-1, 1], [-1,-1,-1]],
                       dtype=np.float64) * np.array([dx/2, dy/2, dz/2])
    corners = (R @ offsets.T).T + np.array([cx, cy, cz])
    return corners


BOX_EDGES = [(0,1),(2,3),(4,5),(6,7),
             (0,2),(1,3),(4,6),(5,7),
             (0,4),(1,5),(2,6),(3,7)]

BOX_BOTTOM = [0,2,3,1]   # bottom face indices (z-)
BOX_TOP    = [4,6,7,5]


def transform_pts(T_inv, pts_world):
    ones = np.ones((len(pts_world), 1))
    return (T_inv @ np.hstack([pts_world, ones]).T).T[:, :3]


# ── Road surface ─────────────────────────────────────────────────────────────

def fill_gradient_bg(canvas):
    """Replace flat BG with vertical gradient: dark sky → dark road surface."""
    h, w = canvas.shape[:2]
    horizon_y = int(h * 0.22)

    sky       = np.array([12, 10,  8], dtype=np.float32)   # near-black (top)
    road_far  = np.array([32, 36, 40], dtype=np.float32)   # dark asphalt (horizon)
    road_near = np.array([42, 46, 50], dtype=np.float32)   # slightly lighter (bottom)

    ys = np.arange(h, dtype=np.float32)
    t_above = np.clip(ys / max(1, horizon_y), 0, 1)
    t_below = np.clip((ys - horizon_y) / max(1, h - horizon_y), 0, 1)
    above = ys < horizon_y

    colors = np.where(above[:, None],
                      sky[None] * (1 - t_above[:, None]) + road_far[None] * t_above[:, None],
                      road_far[None] * (1 - t_below[:, None]) + road_near[None] * t_below[:, None]
                      ).astype(np.uint8)
    canvas[:] = colors[:, None, :]


def draw_road(canvas, proj, view_range, back_range, road_width):
    """Subtle road polygon over gradient background."""
    hw = road_width / 2
    corners_3d = np.array([
        [ view_range, -hw, 0],
        [ view_range,  hw, 0],
        [-back_range,  hw, 0],
        [-back_range, -hw, 0],
    ])
    pts_2d, _ = proj(corners_3d)
    # Only fill if all corners project in front (positive depth handled by clamp,
    # so we check screen bounds are reasonable)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts_2d.reshape(-1, 1, 2)], ROAD_COLOR)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)


# ── Lane lines ───────────────────────────────────────────────────────────────

def draw_lanes(canvas, proj, lane_pts_world, T_inv, view_range, back_range):
    for lane_w in lane_pts_world:
        pts_ego = transform_pts(T_inv, lane_w)
        # Keep only points in visible range
        mask = ((pts_ego[:, 0] >= -back_range) & (pts_ego[:, 0] <= view_range) &
                (np.abs(pts_ego[:, 1]) <= view_range))
        pts_ego = pts_ego[mask]
        if len(pts_ego) < 2:
            continue
        # Sort by forward distance for correct ordering
        order = np.argsort(pts_ego[:, 0])
        pts_ego = pts_ego[order]
        scr, _ = proj(pts_ego)
        pts_cv = scr.reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_cv], False, LANE_COLOR,
                      thickness=LANE_THICKNESS, lineType=cv2.LINE_AA)


# ── 3D boxes ─────────────────────────────────────────────────────────────────

def draw_box(canvas, proj, box, color, edge_color):
    corners = box_corners(box)
    scr, depth = proj(corners)

    # Fill bottom face (road level) with semi-transparent color
    bottom = scr[BOX_BOTTOM]
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [bottom.reshape(-1, 1, 2)], color)
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

    # Draw all 12 edges
    for i, j in BOX_EDGES:
        p1 = tuple(scr[i])
        p2 = tuple(scr[j])
        cv2.line(canvas, p1, p2, edge_color, BOX_THICKNESS, cv2.LINE_AA)


def draw_boxes(canvas, proj, boxes, score_thresh=0.5):
    if boxes is None or len(boxes) == 0:
        return
    for box in boxes:
        if len(box) >= 8 and box[7] < score_thresh:  # score field if present
            continue
        draw_box(canvas, proj, box[:7], BOX_COLOR, BOX_EDGE_COLOR)


# ── Ego vehicle ───────────────────────────────────────────────────────────────

def draw_ego(canvas, proj):
    ego_box = np.array([2.5, 0.0, 0.8, 4.5, 2.0, 1.6, 0.0])
    draw_box(canvas, proj, ego_box, EGO_COLOR, (255, 220, 50))


# ── Frame ID / HUD ────────────────────────────────────────────────────────────

def draw_hud(canvas, frame_id, n_boxes):
    cv2.putText(canvas, f'Frame {frame_id}', (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
    cv2.putText(canvas, f'{n_boxes} vehicles', (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene',       default='highway_sunny_day_2026-04-20-12-58-47')
    parser.add_argument('--out',         default=None)
    parser.add_argument('--fps',         type=int,   default=10)
    parser.add_argument('--elev',        type=float, default=ELEV)
    parser.add_argument('--score_thresh',type=float, default=0.4)
    parser.add_argument('--start_frame', default=None)
    parser.add_argument('--end_frame',   default=None)
    args = parser.parse_args()

    scene_dir  = os.path.join(REPO, 'data', 'roadlane', '0429', args.scene)
    lane_json  = os.path.join(REPO, 'lane_line', 'output', 'fitted',
                              args.scene, 'fitted_lanes.json')
    box_pkl    = os.path.join(scene_dir, '3dbox_result.pkl')
    pose_pkl   = os.path.join(REPO, 'occupancy', 'g6', 'cvpr_format_occ_gen_g6',
                              'output', args.scene, 'pose_dict.pkl')

    out_path = args.out or os.path.join(REPO, 'video_out',
                                         f'{args.scene}_fsd.mp4')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load data
    with open(lane_json) as f:
        lanes = json.load(f)
    lane_pts_world = [np.array(l['points']) for l in lanes]
    print(f'Lane lines: {len(lane_pts_world)}')

    with open(box_pkl, 'rb') as f:
        box_data = pickle.load(f)
    box_by_frame = {str(int(b['frame_id'])): b for b in box_data}
    print(f'3D box frames: {len(box_by_frame)}')

    with open(pose_pkl, 'rb') as f:
        pose_dict = pickle.load(f)

    frame_ids = sorted(pose_dict.keys())
    if args.start_frame:
        frame_ids = [f for f in frame_ids if f >= args.start_frame]
    if args.end_frame:
        frame_ids = [f for f in frame_ids if f <= args.end_frame]
    print(f'Rendering {len(frame_ids)} frames → {out_path}')

    proj = make_proj(args.elev, W, H)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (W, H))

    for fid in frame_ids:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        fill_gradient_bg(canvas)

        T_inv = np.linalg.inv(pose_dict[fid]['matrix'])

        # Road
        draw_road(canvas, proj, VIEW_RANGE, BACK_RANGE, ROAD_WIDTH)

        # Lane lines (world → ego)
        draw_lanes(canvas, proj, lane_pts_world, T_inv, VIEW_RANGE, BACK_RANGE)

        # 3D boxes (already in ego/LiDAR frame)
        frame_idx = str(int(fid))
        boxes = None
        n_boxes = 0
        if frame_idx in box_by_frame:
            b = box_by_frame[frame_idx]
            mask = b['score'] >= args.score_thresh
            boxes = b['boxes_lidar'][mask]
            n_boxes = len(boxes)
            draw_boxes(canvas, proj, boxes)

        # Ego
        draw_ego(canvas, proj)

        # HUD
        draw_hud(canvas, fid, n_boxes)

        writer.write(canvas)

    writer.release()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
