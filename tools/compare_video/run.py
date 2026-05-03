#!/usr/bin/env python3
"""
run.py — CLI entry point for the universal occupancy comparison video tool.

Supports three camera modes:
  --cam_mode none      : no camera strip
  --cam_mode single    : one camera directory (ELAN-style)
  --cam_mode 6cam_dir  : six camera sub-directories under a scene dir (U5-style)
  --cam_mode nuscenes  : read cameras via NuScenes API

Examples
--------
# ELAN (single camera)
python3 tools/compare_video/run.py \
    --dirs       cvpr_format_occ_gen_elan/output/citystreet.../raw \
                 cvpr_format_occ_gen_elan/output/citystreet.../heuristic \
                 cvpr_format_occ_gen_elan/output/citystreet.../seg \
    --labels     "Raw" "Heuristic" "Semantic" \
    --bev_dir    cvpr_format_occ_gen_elan/output/citystreet.../seg \
    --cam_mode   single \
    --cam_dir    data/elan/citystreet.../image \
    --out        video_out/elan_compare.mp4

# U5 (6 cameras, directory layout)
python3 tools/compare_video/run.py \
    --dirs       cvpr_format_occ_gen_u5/output/test_.../raw \
                 cvpr_format_occ_gen_u5/output/test_.../heuristic \
                 cvpr_format_occ_gen_u5/output/test_.../semantic \
    --labels     "Raw" "Heuristic" "Semantic" \
    --bev_dir    cvpr_format_occ_gen_u5/output/test_.../semantic \
    --cam_mode   6cam_dir \
    --cam_dir    data/u5/test_... \
    --out        video_out/u5_compare.mp4

# NuScenes
python3 tools/compare_video/run.py \
    --dirs       cvpr_format_occ_gen_v3/output/gt_pose/scene-0061 \
                 cvpr_format_occ_gen_v4/output/kiss_slam_all10/kiss_slam/scene-0061 \
                 cvpr_format_occ_gen_v4/output/kiss_slam/scene-0061 \
    --labels     "GT pose" "KISS-SLAM all=10" "KISS-SLAM road=40" \
    --bev_dir    cvpr_format_occ_gen_v4/output/kiss_slam/scene-0061 \
    --cam_mode   nuscenes \
    --scene      scene-0061 \
    --out        video_out/nuscenes_compare.mp4
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from tools.compare_video.renderer import render_comparison_video, render_three_view_video

# U5 six-camera directory mapping
U5_CAM_DIRS = {
    'FRONT_LEFT':  ['port_2_camera', 'left'],
    'FRONT':       ['port_8_camera', 'cam_front', 'main'],
    'FRONT_RIGHT': ['port_5_camera', 'right'],
    'BACK_LEFT':   ['port_3_camera', 'sideL'],
    'BACK':        ['port_7_camera', 'cam_back', 'rear'],
    'BACK_RIGHT':  ['port_6_camera', 'sideR'],
}


def make_single_cam(cam_dir, ext='.jpg'):
    """get_cameras for single front-camera directory (frame_id as filename)."""
    def get_cameras(frame_id):
        path = os.path.join(cam_dir, f'{frame_id}{ext}')
        return {'FRONT': path}
    return get_cameras


def make_6cam_dir(scene_dir, ext='.jpg'):
    """
    get_cameras for U5-style layout:
      scene_dir/
        port_2_camera/000000.jpg   (front left)
        port_8_camera/000000.jpg   (front)
        ...
    """
    def get_cameras(frame_id):
        result = {}
        for key, dir_candidates in U5_CAM_DIRS.items():
            for sub in dir_candidates:
                p = os.path.join(scene_dir, sub, f'{frame_id}{ext}')
                if os.path.exists(p):
                    result[key] = p
                    break
        return result
    return get_cameras


def make_nuscenes_cam(nusc, scene_name):
    """get_cameras for NuScenes: maps sample_token -> 6 camera paths."""
    from nuscenes.nuscenes import NuScenes

    CAM_MAP = {
        'FRONT_LEFT':  'CAM_FRONT_LEFT',
        'FRONT':       'CAM_FRONT',
        'FRONT_RIGHT': 'CAM_FRONT_RIGHT',
        'BACK_LEFT':   'CAM_BACK_LEFT',
        'BACK':        'CAM_BACK',
        'BACK_RIGHT':  'CAM_BACK_RIGHT',
    }

    # Build token -> paths dict upfront
    token_cam = {}
    scene = next(s for s in nusc.scene if s['name'] == scene_name)
    tok = scene['first_sample_token']
    while tok:
        sample = nusc.get('sample', tok)
        paths = {}
        for key, nusc_key in CAM_MAP.items():
            sd = nusc.get('sample_data', sample['data'][nusc_key])
            paths[key] = os.path.join(nusc.dataroot, sd['filename'])
        token_cam[tok] = paths
        tok = sample['next'] if sample['next'] else None

    def get_cameras(frame_id):
        return token_cam.get(frame_id, {})
    return get_cameras


def main():
    parser = argparse.ArgumentParser(description='Universal occupancy comparison video')
    parser.add_argument('--dirs',     nargs='+', required=True,
                        help='One occupancy output dir per panel (scene level)')
    parser.add_argument('--labels',   nargs='+', required=True,
                        help='Label for each panel (same count as --dirs)')
    parser.add_argument('--bev_dir',  required=True,
                        help='Occupancy dir used for BEV panel')
    parser.add_argument('--out',      default=None,
                        help='Output .mp4 path (auto-generated if not set)')

    # Camera options
    parser.add_argument('--cam_mode', default='auto',
                        choices=['auto', 'none', 'single', '6cam_dir', 'nuscenes'],
                        help='Camera source type (auto: detect from cam_dir)')
    parser.add_argument('--cam_dir',  default=None,
                        help='Camera directory (single or 6cam_dir mode)')
    parser.add_argument('--cam_ext',  default='.jpg',
                        help='Camera image extension (default: .jpg)')

    # NuScenes options
    parser.add_argument('--scene',    default='scene-0061')
    parser.add_argument('--dataroot', default='/data2/t113c52027/occ_gt_v2/data/nuscenes_occ')
    parser.add_argument('--version',  default='v1.0-mini')

    # Layout options
    parser.add_argument('--start_frame', type=str, default=None,
                        help='First frame id to include (e.g. 000005)')
    parser.add_argument('--end_frame',   type=str, default=None,
                        help='Last frame id to include (e.g. 000060)')
    parser.add_argument('--fps',       type=int,   default=10)
    parser.add_argument('--elev',      type=float, default=28)
    parser.add_argument('--total_w',   type=int,   default=1920)
    parser.add_argument('--total_h',   type=int,   default=1080)
    parser.add_argument('--cam_h',     type=int,   default=500)
    parser.add_argument('--bev_col_w', type=int,   default=500)
    parser.add_argument('--grid_z',      type=int,   default=20,
                        help='Z layers of occupancy grid (16=NuScenes, 20=U5/ELAN)')
    parser.add_argument('--voxel_style', default='flat', choices=['flat', 'shaded'],
                        help='Voxel render style: flat=uniform brightness, shaded=edge darkening')
    parser.add_argument('--z_offset', type=float, default=None,
                        help='Z offset for voxel coords (default: auto from grid_z, e.g. -2.0 for NuScenes, -3.0 for G6)')

    # Fitted lane overlay (independent of occupancy npz)
    parser.add_argument('--lane_json', default=None,
                        help='Path to fitted_lanes.json for on-the-fly BEV overlay')
    parser.add_argument('--pose_pkl',  default=None,
                        help='Path to pose_dict.pkl for lane world→ego projection')

    # Three-view layout (Main 3D | 360 BEV | FSD)
    parser.add_argument('--three_view', action='store_true',
                        help='Use Main/360/FSD layout instead of side-by-side comparison')
    parser.add_argument('--box_pkl',   default=None,
                        help='Path to 3dbox_result.pkl for FSD panel boxes')
    parser.add_argument('--score_thresh', type=float, default=0.4,
                        help='Score threshold for 3D box detections (FSD panel)')
    parser.add_argument('--fsd_elev',  type=float, default=14.0,
                        help='Camera elevation angle (deg) for FSD panel')

    args = parser.parse_args()

    if not args.three_view and len(args.dirs) != len(args.labels):
        parser.error('--dirs and --labels must have the same number of entries')

    # Collect frame IDs from first dir
    first_dir = args.dirs[0]
    if args.cam_mode == 'nuscenes':
        # Use NuScenes API to get correct temporal order
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=args.version, dataroot=args.dataroot)
        scene = next(s for s in nusc.scene if s['name'] == args.scene)
        tok = scene['first_sample_token']
        frame_ids = []
        while tok:
            sample = nusc.get('sample', tok)
            frame_ids.append(tok)
            tok = sample['next'] if sample['next'] else None
        # Filter to only frames that exist in first_dir
        frame_ids = [f for f in frame_ids
                     if os.path.exists(os.path.join(first_dir, f, 'labels.npz'))]
    else:
        frame_ids = sorted([f for f in os.listdir(first_dir)
                            if os.path.exists(os.path.join(first_dir, f, 'labels.npz'))])
    if not frame_ids:
        print(f"ERROR: no labels.npz found in {first_dir}")
        return
    if args.start_frame:
        frame_ids = [f for f in frame_ids if f >= args.start_frame]
    if args.end_frame:
        frame_ids = [f for f in frame_ids if f <= args.end_frame]
    print(f"Found {len(frame_ids)} frames in {first_dir}")

    # Auto-detect cam_mode from cam_dir
    if args.cam_mode == 'auto':
        if not args.cam_dir:
            args.cam_mode = 'none'
        else:
            # Check if cam_dir contains images directly (single) or subdirectories (6cam_dir)
            entries = os.listdir(args.cam_dir) if os.path.isdir(args.cam_dir) else []
            has_images = any(e.endswith(args.cam_ext) for e in entries)
            has_subdirs = any(os.path.isdir(os.path.join(args.cam_dir, e)) for e in entries
                              if e in [d for cands in U5_CAM_DIRS.values() for d in cands])
            if has_subdirs:
                args.cam_mode = '6cam_dir'
            elif has_images:
                args.cam_mode = 'single'
            else:
                args.cam_mode = 'none'
        print(f"[auto] cam_mode detected: {args.cam_mode}")

    # Camera source
    if args.cam_mode == 'none':
        get_cameras = lambda fid: {}
    elif args.cam_mode == 'single':
        if not args.cam_dir:
            parser.error('--cam_dir required for single mode')
        get_cameras = make_single_cam(args.cam_dir, args.cam_ext)
    elif args.cam_mode == '6cam_dir':
        if not args.cam_dir:
            parser.error('--cam_dir required for 6cam_dir mode')
        get_cameras = make_6cam_dir(args.cam_dir, args.cam_ext)
    elif args.cam_mode == 'nuscenes':
        if 'nusc' not in dir():
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes(version=args.version, dataroot=args.dataroot)
        get_cameras = make_nuscenes_cam(nusc, args.scene)

    # Auto-generate output path if not specified
    if args.out is None:
        # Infer dataset and scene from first dir path
        parts = os.path.normpath(args.dirs[0]).split(os.sep)
        # Find the part containing dataset keyword
        dataset = 'unknown'
        for p in parts:
            if 'elan' in p.lower():
                dataset = 'elan'; break
            if 'u5' in p.lower():
                dataset = 'u5'; break
            if 'nuscenes' in p.lower() or 'cvpr_format_occ_gen' in p.lower():
                dataset = 'nuscenes'; break
        # Scene name is the directory just above the mode (raw/heuristic/seg/semantic)
        # structure: .../output/<scene>/<mode>  or  .../output/<backend>/<scene>
        scene_name = args.scene if args.cam_mode == 'nuscenes' else parts[-2]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video_out')
        os.makedirs(out_dir, exist_ok=True)
        args.out = os.path.join(out_dir, f'{dataset}_{scene_name}.mp4')
    print(f"Output: {args.out}")

    grid_shape = (200, 200, args.grid_z)

    # Auto-compute z_offset from grid_z if not specified:
    # z_offset = z_min of the scene, derived as: grid is [0..grid_z-1], voxel=0.4
    # NuScenes: grid_z=16, z range [-5, 3) => z_offset=-5 + some shift => default -2.0
    # G6/ELAN:  grid_z=21, z range [-3, 5.4) => z_offset=-3.0
    # U5:       grid_z=20, z range same as ELAN
    if args.z_offset is not None:
        z_offset = args.z_offset
    elif args.grid_z == 16:
        z_offset = -2.0   # NuScenes
    elif args.grid_z == 21:
        z_offset = -3.0   # G6
    else:
        z_offset = -2.0   # default (U5/ELAN grid_z=20)

    # Load fitted lane lines (optional, independent of occupancy)
    import numpy as np
    lane_pts_world = None
    pose_dict_lane = None
    if args.lane_json and os.path.exists(args.lane_json):
        import json, pickle
        with open(args.lane_json) as f:
            lanes = json.load(f)
        lane_pts_world = []
        for lane in lanes:
            lane_pts_world.append(np.array(lane['points'], dtype=np.float64))
        print(f'Loaded {len(lane_pts_world)} fitted lanes from {args.lane_json}')
        if args.pose_pkl and os.path.exists(args.pose_pkl):
            with open(args.pose_pkl, 'rb') as f:
                pose_dict_lane = pickle.load(f)
            print(f'Loaded pose_dict ({len(pose_dict_lane)} frames) from {args.pose_pkl}')

    # Three-view layout: Main | 360 | FSD
    if args.three_view:
        if len(args.dirs) < 2:
            parser.error('--three_view requires --dirs to provide TWO occupancy dirs '
                         '(main_seg_dir 360_seg_dir)')
        box_by_frame = None
        if args.box_pkl and os.path.exists(args.box_pkl):
            import pickle
            with open(args.box_pkl, 'rb') as f:
                box_data = pickle.load(f)
            box_by_frame = {str(int(b['frame_id'])): b for b in box_data}
            print(f'Loaded {len(box_by_frame)} box frames from {args.box_pkl}')

        # Three-view defaults to shaded voxels unless user overrode it
        voxel_style = 'shaded' if args.voxel_style == 'flat' else args.voxel_style

        render_three_view_video(
            frame_ids      = frame_ids,
            main_occ_dir   = args.dirs[0],
            occ_360_dir    = args.dirs[1],
            get_cameras    = get_cameras,
            out_path       = args.out,
            box_by_frame   = box_by_frame,
            score_thresh   = args.score_thresh,
            lane_pts_world = lane_pts_world,
            pose_dict      = pose_dict_lane,
            fps            = args.fps,
            elev           = args.elev,
            fsd_elev       = args.fsd_elev,
            total_w        = args.total_w,
            total_h        = args.total_h,
            cam_h          = args.cam_h,
            grid_shape     = grid_shape,
            voxel_style    = voxel_style,
            z_offset       = z_offset,
        )
        return

    render_comparison_video(
        frame_ids      = frame_ids,
        occ_dirs       = args.dirs,
        panel_labels   = args.labels,
        bev_dir        = args.bev_dir,
        get_cameras    = get_cameras,
        out_path       = args.out,
        fps            = args.fps,
        elev           = args.elev,
        total_w        = args.total_w,
        total_h        = args.total_h,
        cam_h          = args.cam_h,
        bev_col_w      = args.bev_col_w,
        grid_shape     = grid_shape,
        voxel_style    = args.voxel_style,
        z_offset       = z_offset,
        lane_pts_world = lane_pts_world,
        pose_dict      = pose_dict_lane,
    )


if __name__ == '__main__':
    main()
