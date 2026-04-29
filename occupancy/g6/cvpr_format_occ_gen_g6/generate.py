import os
import glob
import json
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import argparse
import sys
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pose_backends'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'elan', 'cvpr_format_occ_gen_elan'))
from elan_box_utils import fill_box_interior, points_in_boxes

GT_BOUNDS = [-40.0, -40.0, -3.0, 40.0, 40.0, 5.4]
GT_VOXEL  = 0.4
GT_GRID   = (200, 200, 21)

# boxes_lidar Z is in ground-plane frame (Z=0=ground),
# but point cloud Z=0 is the LiDAR sensor (~1.8m above ground)
BOX_Z_SHIFT = -1.8
EGO_BOX_RADIUS = 4.0   # metres — boxes within this XY radius from origin = ego vehicle

LBL_FREE    = 17
LBL_MANMADE = 15
LBL_ROAD    = 11
LBL_LANE    = 18  # new: lane line

G6_COLOR_TABLE = {
    (215, 150, 248): 11,
    (135, 206, 235): 17,
    (247, 206,  70): 15,
    ( 76,   0,  75):  7,
    ( 55,  55, 255):  6,
    (152, 251, 152):  2,
    (255,  91,  33): 10,
    (220,  20,  60):  4,
    (135,  61,   0):  6,
    ( 43, 191, 235):  2,
}

_G6_PALETTE = np.array(list(G6_COLOR_TABLE.keys()), dtype=np.float32)
_G6_LABELS  = np.array(list(G6_COLOR_TABLE.values()), dtype=np.uint8)


def rgb_to_occ3d_label(colors_float):
    rgb255 = (colors_float * 255).astype(np.float32)
    dists = np.linalg.norm(rgb255[:, None, :] - _G6_PALETTE[None, :, :], axis=2)
    nearest_idx = np.argmin(dists, axis=1)
    return _G6_LABELS[nearest_idx]


def _read_pcd_xyz(path):
    """Fast binary PCD reader — returns Nx3 float32 xyz only."""
    with open(path, 'rb') as f:
        header = {}
        while True:
            raw = f.readline()
            if not raw:
                return np.zeros((0, 3), dtype=np.float32)
            line = raw.decode('utf-8', errors='ignore').strip()
            if line.startswith('DATA'):
                data_type = line.split()[1]
                break
            if line.startswith('FIELDS'):
                header['fields'] = line.split()[1:]
            elif line.startswith('SIZE'):
                header['size'] = [int(x) for x in line.split()[1:]]
            elif line.startswith('TYPE'):
                header['type'] = line.split()[1:]
            elif line.startswith('COUNT'):
                header['count'] = [int(x) for x in line.split()[1:]]
            elif line.startswith('POINTS'):
                header['points'] = int(line.split()[1])
        if data_type == 'binary':
            fields = header['fields']
            sizes  = header['size']
            types  = header['type']
            counts = header['count']
            # Build numpy dtype
            dt_map = {'F': 'f', 'I': 'i', 'U': 'u'}
            dtype_list = []
            for f_name, sz, tp, cnt in zip(fields, sizes, types, counts):
                for c in range(cnt):
                    dtype_list.append((f'{f_name}_{c}' if cnt > 1 else f_name,
                                       f'{dt_map[tp]}{sz}'))
            dtype = np.dtype(dtype_list)
            n = header['points']
            data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype)
            return np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float64)
        else:
            # ASCII fallback
            lines = f.read().decode().strip().split('\n')
            pts = np.array([[float(v) for v in l.split()[:3]] for l in lines], dtype=np.float64)
            return pts


def _read_pcd_xyz_rgb(path):
    """Fast binary PCD reader for semantic PCD — returns (Nx3 xyz, Nx3 rgb float[0,1])."""
    import struct
    if os.path.getsize(path) == 0:
        print(f"  [WARN] Empty PCD file, skipping: {path}")
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float32)
    with open(path, 'rb') as f:
        header = {}
        while True:
            raw = f.readline()
            if not raw:
                return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float32)
            line = raw.decode('utf-8', errors='ignore').strip()
            if line.startswith('DATA'):
                data_type = line.split()[1]
                break
            if line.startswith('FIELDS'):
                header['fields'] = line.split()[1:]
            elif line.startswith('SIZE'):
                header['size'] = [int(x) for x in line.split()[1:]]
            elif line.startswith('TYPE'):
                header['type'] = line.split()[1:]
            elif line.startswith('COUNT'):
                header['count'] = [int(x) for x in line.split()[1:]]
            elif line.startswith('POINTS'):
                header['points'] = int(line.split()[1])
        fields = header.get('fields', [])
        if data_type == 'binary':
            sizes  = header['size']
            types  = header['type']
            counts = header['count']
            dt_map = {'F': 'f', 'I': 'i', 'U': 'u'}
            dtype_list = []
            for f_name, sz, tp, cnt in zip(fields, sizes, types, counts):
                for c in range(cnt):
                    dtype_list.append((f'{f_name}_{c}' if cnt > 1 else f_name,
                                       f'{dt_map[tp]}{sz}'))
            dtype = np.dtype(dtype_list)
            n = header['points']
            data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype)
            xyz = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float64)
            # rgb packed as float32 — unpack to 3 bytes
            if 'rgb' in fields:
                rgb_packed = data['rgb'].view(np.uint32)
                r = ((rgb_packed >> 16) & 0xFF).astype(np.float32) / 255.0
                g = ((rgb_packed >>  8) & 0xFF).astype(np.float32) / 255.0
                b = ( rgb_packed        & 0xFF).astype(np.float32) / 255.0
                rgb = np.stack([r, g, b], axis=1)
            else:
                rgb = np.zeros((len(xyz), 3), dtype=np.float32)
            return xyz, rgb
        else:
            lines = f.read().decode().strip().split('\n')
            rows = [l.split() for l in lines]
            xyz = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows], dtype=np.float64)
            rgb = np.zeros((len(xyz), 3), dtype=np.float32)
            return xyz, rgb


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── Worker global state (initialized once per worker process) ──
_W_FRAMES     = None
_W_FILES      = None   # pcd_files or sem_files
_W_POSE_DICT  = None
_W_BOX_DICT   = None
_W_NUM_SWEEPS = None
_W_OUT_DIR    = None
_W_PRELOADED  = None   # preloaded point clouds list (xyz arrays)
_W_LANE_DIR   = None   # path to per-frame lane JSON folder (optional)


def _worker_init(frames, files, pose_dict, box_dict, num_sweeps, out_dir, preloaded=None, lane_dir=None):
    global _W_FRAMES, _W_FILES, _W_POSE_DICT, _W_BOX_DICT, _W_NUM_SWEEPS, _W_OUT_DIR, _W_PRELOADED, _W_LANE_DIR
    _W_FRAMES     = frames
    _W_FILES      = files
    _W_POSE_DICT  = pose_dict
    _W_BOX_DICT   = box_dict
    _W_NUM_SWEEPS = num_sweeps
    _W_OUT_DIR    = out_dir
    _W_PRELOADED  = preloaded
    _W_LANE_DIR   = lane_dir


def _load_lane_pts(json_path):
    """Load lane points (vehicle frame [x_forward, y_lateral, z]) from JSON."""
    if not os.path.exists(json_path):
        return np.zeros((0, 3))
    with open(json_path) as f:
        d = json.load(f)
    pts_list = []
    if 'lane_lines' in d:  # 0413 format
        for lane in d['lane_lines']:
            y = np.array(lane['xyz'][0], dtype=np.float64)
            x = np.array(lane['xyz'][1], dtype=np.float64)
            z = np.array(lane['xyz'][2], dtype=np.float64)
            pts_list.append(np.stack([x, y, z], axis=1))
    elif 'xyz' in d:       # 0429 format
        xyz = d['xyz']
        if xyz[0]:
            lateral = np.array(xyz[0], dtype=np.float64)
            forward = np.array(xyz[1], dtype=np.float64)
            z       = np.array(xyz[2], dtype=np.float64)
            pts_list.append(np.stack([forward, lateral, z], axis=1))
    return np.vstack(pts_list) if pts_list else np.zeros((0, 3))


def _inject_lane(occ, frame_idx, frames, curr_pose_inv):
    """Accumulate lane points from ±num_sweeps frames and write LBL_LANE into occ."""
    if _W_LANE_DIR is None:
        return
    min_b = np.array(GT_BOUNDS[:3])
    max_b = np.array(GT_BOUNDS[3:])
    pose_dict  = _W_POSE_DICT
    num_sweeps = _W_NUM_SWEEPS
    sweep_range = range(max(0, frame_idx - num_sweeps),
                        min(len(frames), frame_idx + num_sweeps + 1))
    for j in sweep_range:
        fid  = frames[j]
        pts  = _load_lane_pts(os.path.join(_W_LANE_DIR, f'{fid}.json'))
        if not len(pts):
            continue
        if j != frame_idx:
            p_homo = np.hstack([pts, np.ones((len(pts), 1))])
            pts = (curr_pose_inv @ pose_dict[fid]['matrix'] @ p_homo.T).T[:, :3]
        bm = np.all((pts >= min_b) & (pts < max_b), axis=1)
        if not np.any(bm):
            continue
        idxs = np.clip(((pts[bm] - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
        # Fill ground layer + 2 layers above to make lane lines visible in 3D
        for dz in range(3):
            iz = np.clip(idxs[:, 2] + dz, 0, GT_GRID[2] - 1)
            occ[idxs[:, 0], idxs[:, 1], iz] = LBL_LANE


# ── SEG worker ──
def _seg_worker(i):
    DYNAMIC_LABELS = {2, 4, 6, 7, 10}
    frames    = _W_FRAMES
    sem_files = _W_FILES
    pose_dict = _W_POSE_DICT
    box_dict  = _W_BOX_DICT
    num_sweeps = _W_NUM_SWEEPS
    out_dir   = _W_OUT_DIR

    frame_id      = frames[i]
    min_b         = np.array(GT_BOUNDS[:3])
    max_b         = np.array(GT_BOUNDS[3:])
    curr_pose     = pose_dict[frame_id]['matrix']
    curr_pose_inv = np.linalg.inv(curr_pose)
    curr_boxes    = box_dict.get(frame_id, {'names': [], 'boxes_lidar': []})
    occ = np.ones(GT_GRID, dtype=np.uint8) * LBL_FREE

    def load_sem(idx, remove_dynamic=True):
        pts, cols = _read_pcd_xyz_rgb(sem_files[idx])
        if len(pts) == 0:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.uint8)
        labels = rgb_to_occ3d_label(cols)
        keep = labels != LBL_FREE
        pts, labels = pts[keep], labels[keep]
        if remove_dynamic and len(pts) > 0:
            static = np.array([lbl not in DYNAMIC_LABELS for lbl in labels])
            pts, labels = pts[static], labels[static]
        if len(pts) > 0:
            ego_mask = (
                (np.abs(pts[:, 0]) < 2.5) &
                (np.abs(pts[:, 1]) < 1.2) &
                (pts[:, 2] < 0.0)
            )
            pts, labels = pts[~ego_mask], labels[~ego_mask]
        return pts, labels

    sem_pts, sem_lbl = load_sem(i)
    if len(sem_pts) > 0:
        curr_boxes_lidar = curr_boxes.get('boxes_lidar', [])
        if len(curr_boxes_lidar) > 0:
            dyn = points_in_boxes(sem_pts, curr_boxes_lidar)
            sem_pts, sem_lbl = sem_pts[~dyn], sem_lbl[~dyn]
        mask = np.all((sem_pts >= min_b) & (sem_pts < max_b), axis=1)
        if np.any(mask):
            idxs = np.clip(((sem_pts[mask] - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
            occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = sem_lbl[mask]

    for j in range(1, num_sweeps + 1):
        if i - j < 0: break
        prev_id   = frames[i - j]
        prev_pose = pose_dict[prev_id]['matrix']
        pts, lbl  = load_sem(i - j)
        if len(pts) == 0: continue
        prev_boxes = box_dict.get(prev_id, {'boxes_lidar': []})['boxes_lidar']
        if len(prev_boxes) > 0:
            dyn = points_in_boxes(pts, prev_boxes)
            pts, lbl = pts[~dyn], lbl[~dyn]
        if len(pts) == 0: continue
        p_homo = np.hstack((pts, np.ones((len(pts), 1))))
        t_pts  = (curr_pose_inv @ prev_pose @ p_homo.T).T[:, :3]
        bm = np.all((t_pts >= min_b) & (t_pts < max_b), axis=1)
        if np.any(bm):
            t_idxs = np.clip(((t_pts[bm] - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
            occ[t_idxs[:, 0], t_idxs[:, 1], t_idxs[:, 2]] = lbl[bm]

    for j in range(1, num_sweeps + 1):
        if i + j >= len(frames): break
        next_id   = frames[i + j]
        next_pose = pose_dict[next_id]['matrix']
        pts, lbl  = load_sem(i + j)
        if len(pts) == 0: continue
        next_boxes = box_dict.get(next_id, {'boxes_lidar': []})['boxes_lidar']
        if len(next_boxes) > 0:
            dyn = points_in_boxes(pts, next_boxes)
            pts, lbl = pts[~dyn], lbl[~dyn]
        if len(pts) == 0: continue
        p_homo = np.hstack((pts, np.ones((len(pts), 1))))
        t_pts  = (curr_pose_inv @ next_pose @ p_homo.T).T[:, :3]
        bm = np.all((t_pts >= min_b) & (t_pts < max_b), axis=1)
        if np.any(bm):
            t_idxs = np.clip(((t_pts[bm] - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
            occ[t_idxs[:, 0], t_idxs[:, 1], t_idxs[:, 2]] = lbl[bm]

    filled = 0
    if len(curr_boxes['names']) > 0:
        _boxes = curr_boxes['boxes_lidar']
        _names = curr_boxes['names']
        _non_ego = np.hypot(_boxes[:, 0], _boxes[:, 1]) > EGO_BOX_RADIUS
        occ, filled = fill_box_interior(occ, _boxes[_non_ego], _names[_non_ego], GT_BOUNDS, GT_VOXEL, GT_GRID)

    _inject_lane(occ, i, frames, curr_pose_inv)

    save_path = os.path.join(out_dir, frame_id)
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, 'labels.npz'), semantics=occ)
    return filled


# ── RAW worker ──
def _raw_worker(i):
    frames    = _W_FRAMES
    pcd_files = _W_FILES
    pose_dict = _W_POSE_DICT
    box_dict  = _W_BOX_DICT
    num_sweeps = _W_NUM_SWEEPS
    out_dir   = _W_OUT_DIR

    frame_id      = frames[i]
    min_b         = np.array(GT_BOUNDS[:3])
    max_b         = np.array(GT_BOUNDS[3:])
    curr_pose     = pose_dict[frame_id]['matrix']
    curr_pose_inv = np.linalg.inv(curr_pose)
    curr_boxes    = box_dict.get(frame_id, {'names': [], 'boxes_lidar': []})

    def get_pts(idx):
        if _W_PRELOADED is not None:
            return _W_PRELOADED[idx].copy()
        return _read_pcd_xyz(pcd_files[idx])

    pcs = [get_pts(i)]

    for j in range(1, num_sweeps + 1):
        if i - j < 0: break
        prev_id   = frames[i - j]
        prev_pose = pose_dict[prev_id]['matrix']
        prev_pts  = get_pts(i - j)
        prev_boxes = box_dict.get(prev_id, {'boxes_lidar': []})['boxes_lidar']
        if len(prev_boxes) > 0:
            prev_pts = prev_pts[~points_in_boxes(prev_pts, prev_boxes)]
        if len(prev_pts) > 0:
            p_homo = np.hstack((prev_pts, np.ones((len(prev_pts), 1))))
            pcs.append((curr_pose_inv @ prev_pose @ p_homo.T).T[:, :3])

    for j in range(1, num_sweeps + 1):
        if i + j >= len(frames): break
        next_id   = frames[i + j]
        next_pose = pose_dict[next_id]['matrix']
        next_pts  = get_pts(i + j)
        next_boxes = box_dict.get(next_id, {'boxes_lidar': []})['boxes_lidar']
        if len(next_boxes) > 0:
            next_pts = next_pts[~points_in_boxes(next_pts, next_boxes)]
        if len(next_pts) > 0:
            p_homo = np.hstack((next_pts, np.ones((len(next_pts), 1))))
            pcs.append((curr_pose_inv @ next_pose @ p_homo.T).T[:, :3])

    all_pts = np.vstack(pcs)
    mask = np.all((all_pts >= min_b) & (all_pts < max_b), axis=1)
    valid_xyz = all_pts[mask]
    idxs = np.clip(((valid_xyz - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
    occ = np.ones(GT_GRID, dtype=np.uint8) * LBL_FREE
    occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = LBL_MANMADE

    filled = 0
    if len(curr_boxes['names']) > 0:
        _boxes = curr_boxes['boxes_lidar']
        _names = curr_boxes['names']
        _non_ego = np.hypot(_boxes[:, 0], _boxes[:, 1]) > EGO_BOX_RADIUS
        occ, filled = fill_box_interior(occ, _boxes[_non_ego], _names[_non_ego], GT_BOUNDS, GT_VOXEL, GT_GRID)

    _inject_lane(occ, i, frames, curr_pose_inv)

    save_path = os.path.join(out_dir, frame_id)
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, 'labels.npz'), semantics=occ)
    return filled


# ── HEURISTIC worker ──
def _heuristic_worker(i):
    import time
    t0 = time.time()
    frames    = _W_FRAMES
    pcd_files = _W_FILES
    pose_dict = _W_POSE_DICT
    box_dict  = _W_BOX_DICT
    num_sweeps = _W_NUM_SWEEPS
    out_dir   = _W_OUT_DIR

    frame_id      = frames[i]
    min_b         = np.array(GT_BOUNDS[:3])
    max_b         = np.array(GT_BOUNDS[3:])
    curr_pose     = pose_dict[frame_id]['matrix']
    curr_pose_inv = np.linalg.inv(curr_pose)
    curr_boxes    = box_dict.get(frame_id, {'names': [], 'boxes_lidar': []})

    def get_pts(idx):
        if _W_PRELOADED is not None:
            return _W_PRELOADED[idx].copy()
        return _read_pcd_xyz(pcd_files[idx])

    pcs = [get_pts(i)]

    for j in range(1, num_sweeps + 1):
        if i - j < 0: break
        prev_id   = frames[i - j]
        prev_pose = pose_dict[prev_id]['matrix']
        prev_pts  = get_pts(i - j)
        prev_boxes = box_dict.get(prev_id, {'boxes_lidar': []})['boxes_lidar']
        if len(prev_boxes) > 0:
            prev_pts = prev_pts[~points_in_boxes(prev_pts, prev_boxes)]
        if len(prev_pts) > 0:
            p_homo = np.hstack((prev_pts, np.ones((len(prev_pts), 1))))
            pcs.append((curr_pose_inv @ prev_pose @ p_homo.T).T[:, :3])

    for j in range(1, num_sweeps + 1):
        if i + j >= len(frames): break
        next_id   = frames[i + j]
        next_pose = pose_dict[next_id]['matrix']
        next_pts  = get_pts(i + j)
        next_boxes = box_dict.get(next_id, {'boxes_lidar': []})['boxes_lidar']
        if len(next_boxes) > 0:
            next_pts = next_pts[~points_in_boxes(next_pts, next_boxes)]
        if len(next_pts) > 0:
            p_homo = np.hstack((next_pts, np.ones((len(next_pts), 1))))
            pcs.append((curr_pose_inv @ next_pose @ p_homo.T).T[:, :3])

    all_pts = np.vstack(pcs)
    mask = np.all((all_pts >= min_b) & (all_pts < max_b), axis=1)
    valid_xyz = all_pts[mask]
    idxs = np.clip(((valid_xyz - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
    occ = np.ones(GT_GRID, dtype=np.uint8) * LBL_FREE
    z_real = valid_xyz[:, 2]
    lbl = np.full(len(z_real), LBL_MANMADE, dtype=np.uint8)
    lbl[z_real < -1.5] = LBL_ROAD
    occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = lbl

    filled = 0
    if len(curr_boxes['names']) > 0:
        _boxes = curr_boxes['boxes_lidar']
        _names = curr_boxes['names']
        _non_ego = np.hypot(_boxes[:, 0], _boxes[:, 1]) > EGO_BOX_RADIUS
        occ, filled = fill_box_interior(occ, _boxes[_non_ego], _names[_non_ego], GT_BOUNDS, GT_VOXEL, GT_GRID)

    _inject_lane(occ, i, frames, curr_pose_inv)

    save_path = os.path.join(out_dir, frame_id)
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, 'labels.npz'), semantics=occ)
    print(f"  [heuristic] frame {i} done in {time.time()-t0:.1f}s", flush=True)
    return filled


# ─────────────────────────────────────────────
# Public generate functions
# ─────────────────────────────────────────────

def _run_pool(worker_fn, init_args, n_frames, num_workers, desc, preload_pcd=False, lane_dir=None):
    frames, files, pose_dict, box_dict, num_sweeps, out_dir = init_args

    preloaded = None
    if preload_pcd:
        print(f"  Preloading {len(files)} PCD files into memory...", flush=True)
        preloaded = [_read_pcd_xyz(f) for f in tqdm(files, desc='Preload')]
        print(f"  Preload done. {sum(p.nbytes for p in preloaded)/1e6:.0f} MB", flush=True)

    print(f"  Spawning {num_workers} workers (fork)...", flush=True)
    ctx = mp.get_context('fork')
    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(frames, files, pose_dict, box_dict, num_sweeps, out_dir, preloaded, lane_dir),
    ) as pool:
        print(f"  Workers ready, submitting {n_frames} tasks...", flush=True)
        results = list(tqdm(pool.imap(worker_fn, range(n_frames)), total=n_frames, desc=desc))
    return sum(results)


def _get_pcd_dir(dataroot):
    d = os.path.join(dataroot, 'VLS128_pcdnpy')
    if not os.path.isdir(d):
        d = os.path.join(dataroot, 'VLS128_pcd')
    return d


def generate_occupancy(dataroot, pose_dict, num_sweeps=40, out_dir='output', num_workers=16, only_frame=None, lane_dir=None, sem_subdir=None):
    pcd_files = sorted(glob.glob(os.path.join(_get_pcd_dir(dataroot), '*.pcd')))
    if sem_subdir:
        sem_dir = os.path.join(dataroot, sem_subdir)
    else:
        for _sem_candidate in ['result_depth_filtered_v2', 'result_depth_filtered', 'colored_360_pcd_filter',
                               'colored_pcd_6view', 'colored_pcd_main']:
            _sem_dir = os.path.join(dataroot, _sem_candidate)
            if os.path.isdir(_sem_dir):
                sem_dir = _sem_dir
                break
    print(f"[INFO] SEG 語意來源：{sem_dir}")
    sem_files = sorted(glob.glob(os.path.join(sem_dir, '*.pcd')))
    n = min(len(pcd_files), len(sem_files))
    sem_files = sem_files[:n]
    print(f"[INFO] SEG 模式：{n} 幀，workers={num_workers}")

    box_data = load_pkl(os.path.join(dataroot, '3dbox_result.pkl'))
    box_dict = {}
    for item in box_data:
        fid = item['file_name'].replace('.pcd', '')
        valid = item['score'] >= 0.4
        boxes = item['boxes_lidar'][valid].copy()
        boxes[:, 2] += BOX_Z_SHIFT
        box_dict[fid] = {
            'names':       item['name'][valid],
            'boxes_lidar': boxes,
            'pred_labels': item['pred_labels'][valid],
        }

    frames = [os.path.splitext(os.path.basename(f))[0] for f in pcd_files[:n]]
    if only_frame is not None:
        idx = [i for i, f in enumerate(frames) if f == only_frame]
        frames = [frames[i] for i in idx]
        sem_files = [sem_files[i] for i in idx]
    os.makedirs(out_dir, exist_ok=True)
    total = _run_pool(_seg_worker, (frames, sem_files, pose_dict, box_dict, num_sweeps, out_dir), len(frames), num_workers, 'SEG', lane_dir=lane_dir)
    print(f"\n[SEG Complete] {total} dynamic voxels filled by 3D Box")


def generate_occupancy_raw(dataroot, pose_dict, num_sweeps=40, out_dir='output', num_workers=16, only_frame=None, lane_dir=None):
    pcd_files = sorted(glob.glob(os.path.join(_get_pcd_dir(dataroot), '*.pcd')))
    print(f"[INFO] RAW 模式：{len(pcd_files)} 幀，workers={num_workers}")

    box_data = load_pkl(os.path.join(dataroot, '3dbox_result.pkl'))
    box_dict = {}
    for item in box_data:
        fid = item['file_name'].replace('.pcd', '')
        valid = item['score'] >= 0.4
        boxes = item['boxes_lidar'][valid].copy()
        boxes[:, 2] += BOX_Z_SHIFT
        box_dict[fid] = {
            'names':       item['name'][valid],
            'boxes_lidar': boxes,
            'pred_labels': item['pred_labels'][valid],
        }

    frames = [os.path.splitext(os.path.basename(f))[0] for f in pcd_files]
    if only_frame is not None:
        idx = [i for i, f in enumerate(frames) if f == only_frame]
        frames = [frames[i] for i in idx]
        pcd_files = [pcd_files[i] for i in idx]
    os.makedirs(out_dir, exist_ok=True)
    total = _run_pool(_raw_worker, (frames, pcd_files, pose_dict, box_dict, num_sweeps, out_dir), len(frames), num_workers, 'RAW', preload_pcd=True, lane_dir=lane_dir)
    print(f"\n[RAW Complete] {total} dynamic voxels filled by 3D Box")


def generate_occupancy_heuristic(dataroot, pose_dict, num_sweeps=40, out_dir='output', num_workers=16, only_frame=None, lane_dir=None):
    pcd_files = sorted(glob.glob(os.path.join(_get_pcd_dir(dataroot), '*.pcd')))
    print(f"[INFO] HEURISTIC 模式：{len(pcd_files)} 幀，workers={num_workers}")

    box_data = load_pkl(os.path.join(dataroot, '3dbox_result.pkl'))
    box_dict = {}
    for item in box_data:
        fid = item['file_name'].replace('.pcd', '')
        valid = item['score'] >= 0.4
        boxes = item['boxes_lidar'][valid].copy()
        boxes[:, 2] += BOX_Z_SHIFT
        box_dict[fid] = {
            'names':       item['name'][valid],
            'boxes_lidar': boxes,
            'pred_labels': item['pred_labels'][valid],
        }

    frames = [os.path.splitext(os.path.basename(f))[0] for f in pcd_files]
    if only_frame is not None:
        idx = [i for i, f in enumerate(frames) if f == only_frame]
        frames = [frames[i] for i in idx]
        pcd_files = [pcd_files[i] for i in idx]
    os.makedirs(out_dir, exist_ok=True)
    total = _run_pool(_heuristic_worker, (frames, pcd_files, pose_dict, box_dict, num_sweeps, out_dir), len(frames), num_workers, 'HEURISTIC', preload_pcd=True, lane_dir=lane_dir)
    print(f"\n[HEURISTIC Complete] {total} dynamic voxels filled by 3D Box")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',     default='kiss_icp_gps')
    parser.add_argument('--scene',       default='citystreet_sunny_day_2026-02-03-15-17-34')
    parser.add_argument('--sweeps',      type=int, default=40)
    parser.add_argument('--mode',        default='all',
                        choices=['semantic', 'raw', 'heuristic', 'all'])
    parser.add_argument('--data_root',   default='/data2/t113c52027/occ_gt_v2/data/g6')
    parser.add_argument('--out_root',    default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--frame',       type=str, default=None,
                        help='Only process this single frame id (e.g. 000044)')
    parser.add_argument('--lane_dir',    type=str, default=None,
                        help='Path to folder with per-frame lane JSON files (optional)')
    parser.add_argument('--sem_subdir',  type=str, default=None,
                        help='Semantic PCD subfolder name (e.g. colored_pcd_main, colored_pcd_6view)')
    parser.add_argument('--out_subdir',  type=str, default=None,
                        help='Override output subdirectory name for semantic mode (e.g. seg_main, seg_360)')
    args = parser.parse_args()

    scene_path = os.path.join(args.data_root, args.scene)

    # Pose cache
    cache_dir = os.path.join(args.out_root, args.scene)
    os.makedirs(cache_dir, exist_ok=True)
    pose_cache = os.path.join(cache_dir, 'pose_dict.pkl')
    if os.path.exists(pose_cache):
        print(f"[INFO] Loading cached pose_dict from {pose_cache}")
        with open(pose_cache, 'rb') as f:
            pose_dict = pickle.load(f)
    else:
        import importlib.util
        backend_path = os.path.join(os.path.dirname(__file__), 'pose_backends', f'{args.backend}.py')
        spec = importlib.util.spec_from_file_location(args.backend, backend_path)
        backend_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend_mod)
        pose_dict = backend_mod.get_pose_dict(scene_path)
        with open(pose_cache, 'wb') as f:
            pickle.dump(pose_dict, f)
        print(f"[INFO] Saved pose_dict to {pose_cache}")

    scene_out = os.path.join(args.out_root, args.scene)

    if args.mode in ('semantic', 'all'):
        seg_subdir = args.out_subdir if args.out_subdir else 'seg'
        generate_occupancy(scene_path, pose_dict, args.sweeps,
                           out_dir=os.path.join(scene_out, seg_subdir),
                           num_workers=args.num_workers, only_frame=args.frame,
                           lane_dir=args.lane_dir, sem_subdir=args.sem_subdir)

    if args.mode in ('raw', 'all'):
        generate_occupancy_raw(scene_path, pose_dict, args.sweeps,
                               out_dir=os.path.join(scene_out, 'raw'),
                               num_workers=args.num_workers, only_frame=args.frame,
                               lane_dir=args.lane_dir)

    if args.mode in ('heuristic', 'all'):
        generate_occupancy_heuristic(scene_path, pose_dict, args.sweeps,
                                     out_dir=os.path.join(scene_out, 'heuristic'),
                                     num_workers=args.num_workers, only_frame=args.frame,
                                     lane_dir=args.lane_dir)


if __name__ == '__main__':
    main()
