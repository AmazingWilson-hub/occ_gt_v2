import os
import glob
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import argparse

import sys
sys.path.insert(0, os.path.dirname(__file__))
from elan_box_utils import fill_box_interior, points_in_boxes

GT_BOUNDS = [-40.0, -40.0, -3.0, 40.0, 40.0, 5.4]  # ELAN: LiDAR 座標系，地面在 z ≈ -2m
GT_VOXEL  = 0.4
GT_GRID   = (200, 200, 21)  # Z 從 -3.0 到 5.4 → 8.4m / 0.4 = 21 layers

# Labels
LBL_FREE = 17
LBL_MANMADE = 15
LBL_ROAD = 11
LBL_SIDEWALK = 13
LBL_TERRAIN = 14

# --- ELAN Semantic Color Table → Occ3D Label ---
# RGB (0-255) → Occ3D label
ELAN_COLOR_TABLE = {
    (215, 150, 248): 11,  # road → driveable_surface
    (135, 206, 235): 17,  # sky → free (ignore)
    (247, 206,  70): 15,  # undriveable area (building, sidewalk) → manmade
    ( 76,   0,  75):  7,  # pedestrian
    ( 55,  55, 255):  6,  # motorcyclist → motorcycle
    (152, 251, 152):  2,  # bicyclist → bicycle
    (255,  91,  33): 10,  # big car → truck
    (220,  20,  60):  4,  # car
    (135,  61,   0):  6,  # motorcycle
    ( 43, 191, 235):  2,  # bicycle
}

# Build lookup array for fast nearest-neighbor matching
_ELAN_PALETTE = np.array(list(ELAN_COLOR_TABLE.keys()), dtype=np.float32)  # (K, 3)
_ELAN_LABELS  = np.array(list(ELAN_COLOR_TABLE.values()), dtype=np.uint8)  # (K,)

def rgb_to_occ3d_label(colors_float):
    """
    Map Nx3 float RGB [0,1] to Occ3D labels via nearest-neighbor to the ELAN palette.
    """
    rgb255 = (colors_float * 255).astype(np.float32)  # (N, 3)
    # Compute L2 distance to each palette color
    dists = np.linalg.norm(rgb255[:, None, :] - _ELAN_PALETTE[None, :, :], axis=2)  # (N, K)
    nearest_idx = np.argmin(dists, axis=1)  # (N,)
    return _ELAN_LABELS[nearest_idx]

def quat_wxyz_to_rot(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def generate_occupancy(dataroot, pose_dict, num_sweeps=40, out_dir='output'):
    pcd_files = sorted(glob.glob(os.path.join(dataroot, 'VLS128_pcdnpy', '*.pcd')))
    sem_files = sorted(glob.glob(os.path.join(dataroot, 'colored_pcd_img', '*.pcd')))
    has_semantic = len(sem_files) == len(pcd_files)
    if not has_semantic:
        print(f"[ERROR] 語意點雲數量 ({len(sem_files)}) 與原始點雲 ({len(pcd_files)}) 不符！無法執行。")
        return
    print(f"[INFO] 新架構：360° LiDAR 只負責定位，語意點雲負責所有 Voxel 填色！")
    
    box_data = load_pkl(os.path.join(dataroot, '3dbox_result.pkl'))
    
    box_dict = {}
    for item in box_data:
        frame_id = item['file_name'].replace('.pcd', '')
        valid = item['score'] >= 0.4
        box_dict[frame_id] = {
            'names': item['name'][valid],
            'boxes_lidar': item['boxes_lidar'][valid],
            'pred_labels': item['pred_labels'][valid]
        }
        
    frames = [os.path.splitext(os.path.basename(f))[0] for f in pcd_files]
    total_filled = 0
    os.makedirs(out_dir, exist_ok=True)
    
    for i, frame_id in enumerate(tqdm(frames, desc='Generating Occupancy for ELAN')):
        
        curr_pose = pose_dict[frame_id]['matrix']
        curr_pose_inv = np.linalg.inv(curr_pose)
        curr_boxes = box_dict.get(frame_id, {'names': [], 'boxes_lidar': []})
        
        min_b = np.array(GT_BOUNDS[:3])
        max_b = np.array(GT_BOUNDS[3:])
        occ = np.ones(GT_GRID, dtype=np.uint8) * LBL_FREE
        
        # ====== 唯一資料來源：語意點雲 (colored_pcd_img) ======
        # 堆疊當前幀 + 前後 N 幀的語意點雲，用 KISS-ICP 的精確軌跡對齊
        
        # 動態物件的 Occ3D 標籤 (由 3D Box 最後填回)
        DYNAMIC_LABELS = {2, 4, 6, 7, 10}  # bicycle, car, motorcycle, pedestrian, truck
        
        def load_and_label_sem(idx, remove_dynamic=True):
            """載入一幀語意點雲，回傳 (xyz, labels)"""
            sem_pcd = o3d.io.read_point_cloud(sem_files[idx])
            pts = np.asarray(sem_pcd.points)
            cols = np.asarray(sem_pcd.colors)
            if len(pts) == 0 or len(cols) == 0:
                return np.zeros((0, 3)), np.zeros(0, dtype=np.uint8)
            labels = rgb_to_occ3d_label(cols)
            # 過濾天空 (label 17)
            keep = labels != LBL_FREE
            pts, labels = pts[keep], labels[keep]
            # 過濾動態物件語意 (車、人、機車等 → 交給 3D Box 填)
            if remove_dynamic and len(pts) > 0:
                static = np.array([lbl not in DYNAMIC_LABELS for lbl in labels])
                pts, labels = pts[static], labels[static]
            # 移除本車車體上的點雲 (LiDAR 原點附近的長方形區域)
            if len(pts) > 0:
                ego_mask = (
                    (np.abs(pts[:, 0]) < 2.5) &
                    (np.abs(pts[:, 1]) < 1.2) &
                    (pts[:, 2] < 0.0)
                )
                pts, labels = pts[~ego_mask], labels[~ego_mask]
            return pts, labels
        
        # 當前幀語意
        sem_pts, sem_lbl = load_and_label_sem(i)
        if len(sem_pts) > 0:
            # 先用 3D Box 挖掉動態物件，讓語意只保留靜態場景
            curr_boxes_lidar = curr_boxes.get('boxes_lidar', [])
            if len(curr_boxes_lidar) > 0:
                dyn = points_in_boxes(sem_pts, curr_boxes_lidar)
                sem_pts, sem_lbl = sem_pts[~dyn], sem_lbl[~dyn]
            mask = np.all((sem_pts >= min_b) & (sem_pts < max_b), axis=1)
            if np.any(mask):
                idxs = np.clip(((sem_pts[mask] - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
                occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = sem_lbl[mask]
        
        # 過去幀語意堆疊
        for j in range(1, num_sweeps + 1):
            if i - j < 0: break
            prev_id = frames[i - j]
            prev_pose = pose_dict[prev_id]['matrix']
            pts, lbl = load_and_label_sem(i - j)
            if len(pts) == 0: continue
            # 去除動態物件殘影
            prev_boxes = box_dict.get(prev_id, {'boxes_lidar': []})['boxes_lidar']
            if len(prev_boxes) > 0:
                dyn = points_in_boxes(pts, prev_boxes)
                pts, lbl = pts[~dyn], lbl[~dyn]
            if len(pts) == 0: continue
            # 用 KISS-ICP 軌跡精準對齊到當前幀
            p_homo = np.hstack((pts, np.ones((len(pts), 1))))
            t_pts = (curr_pose_inv @ prev_pose @ p_homo.T).T[:, :3]
            bm = np.all((t_pts >= min_b) & (t_pts < max_b), axis=1)
            if np.any(bm):
                t_idxs = np.clip(((t_pts[bm] - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
                occ[t_idxs[:, 0], t_idxs[:, 1], t_idxs[:, 2]] = lbl[bm]
        
        # 未來幀語意堆疊
        for j in range(1, num_sweeps + 1):
            if i + j >= len(frames): break
            next_id = frames[i + j]
            next_pose = pose_dict[next_id]['matrix']
            pts, lbl = load_and_label_sem(i + j)
            if len(pts) == 0: continue
            next_boxes = box_dict.get(next_id, {'boxes_lidar': []})['boxes_lidar']
            if len(next_boxes) > 0:
                dyn = points_in_boxes(pts, next_boxes)
                pts, lbl = pts[~dyn], lbl[~dyn]
            if len(pts) == 0: continue
            p_homo = np.hstack((pts, np.ones((len(pts), 1))))
            t_pts = (curr_pose_inv @ next_pose @ p_homo.T).T[:, :3]
            bm = np.all((t_pts >= min_b) & (t_pts < max_b), axis=1)
            if np.any(bm):
                t_idxs = np.clip(((t_pts[bm] - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
                occ[t_idxs[:, 0], t_idxs[:, 1], t_idxs[:, 2]] = lbl[bm]
        
        # ====== 3D Box 體積填充 (全域無死角) ======
        if len(curr_boxes['names']) > 0:
            occ, filled = fill_box_interior(occ, curr_boxes['boxes_lidar'], curr_boxes['names'], GT_BOUNDS, GT_VOXEL, GT_GRID)
            total_filled += filled
            
        save_path = os.path.join(out_dir, frame_id)
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(os.path.join(save_path, 'labels.npz'), semantics=occ)
        
    print(f"\n[V3 Custom Pipeline Complete] 總共有 {total_filled} 個動態物件 Voxel 被 3D Box 霸道填滿實心色彩！")

def generate_occupancy_raw(dataroot, pose_dict, num_sweeps=40, out_dir='output'):
    """純 360° 原始 LiDAR 堆疊版 (無語意)，動態物件用 3D Box 填充"""
    pcd_files = sorted(glob.glob(os.path.join(dataroot, 'VLS128_pcdnpy', '*.pcd')))
    print(f"[INFO] RAW 模式：360° LiDAR 堆疊 + 3D Box 填充 (無語意)")
    
    box_data = load_pkl(os.path.join(dataroot, '3dbox_result.pkl'))
    box_dict = {}
    for item in box_data:
        frame_id = item['file_name'].replace('.pcd', '')
        valid = item['score'] >= 0.4
        box_dict[frame_id] = {
            'names': item['name'][valid],
            'boxes_lidar': item['boxes_lidar'][valid],
            'pred_labels': item['pred_labels'][valid]
        }
    
    frames = [os.path.splitext(os.path.basename(f))[0] for f in pcd_files]
    total_filled = 0
    os.makedirs(out_dir, exist_ok=True)
    
    for i, frame_id in enumerate(tqdm(frames, desc='Generating RAW Occupancy')):
        curr_pose = pose_dict[frame_id]['matrix']
        curr_pose_inv = np.linalg.inv(curr_pose)
        curr_boxes = box_dict.get(frame_id, {'names': [], 'boxes_lidar': []})
        
        min_b = np.array(GT_BOUNDS[:3])
        max_b = np.array(GT_BOUNDS[3:])
        
        # 載入當前幀 360° 點雲
        pcd = o3d.io.read_point_cloud(pcd_files[i])
        curr_pts = np.asarray(pcd.points)
        pcs = [curr_pts]
        
        # 堆疊過去幀 (去除動態物件)
        for j in range(1, num_sweeps + 1):
            if i - j < 0: break
            prev_id = frames[i - j]
            prev_pose = pose_dict[prev_id]['matrix']
            prev_pcd = o3d.io.read_point_cloud(pcd_files[i - j])
            prev_pts = np.asarray(prev_pcd.points)
            prev_boxes = box_dict.get(prev_id, {'boxes_lidar': []})['boxes_lidar']
            if len(prev_boxes) > 0:
                mask = points_in_boxes(prev_pts, prev_boxes)
                prev_pts = prev_pts[~mask]
            if len(prev_pts) > 0:
                p_homo = np.hstack((prev_pts, np.ones((len(prev_pts), 1))))
                transformed = (curr_pose_inv @ prev_pose @ p_homo.T).T[:, :3]
                pcs.append(transformed)
        
        # 堆疊未來幀
        for j in range(1, num_sweeps + 1):
            if i + j >= len(frames): break
            next_id = frames[i + j]
            next_pose = pose_dict[next_id]['matrix']
            next_pcd = o3d.io.read_point_cloud(pcd_files[i + j])
            next_pts = np.asarray(next_pcd.points)
            next_boxes = box_dict.get(next_id, {'boxes_lidar': []})['boxes_lidar']
            if len(next_boxes) > 0:
                mask = points_in_boxes(next_pts, next_boxes)
                next_pts = next_pts[~mask]
            if len(next_pts) > 0:
                p_homo = np.hstack((next_pts, np.ones((len(next_pts), 1))))
                transformed = (curr_pose_inv @ next_pose @ p_homo.T).T[:, :3]
                pcs.append(transformed)
        
        all_pts = np.vstack(pcs)
        
        # 體素化：統一標記為 MANMADE (無語意資訊)
        mask = np.all((all_pts >= min_b) & (all_pts < max_b), axis=1)
        valid_xyz = all_pts[mask]
        idxs = np.clip(((valid_xyz - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
        occ = np.ones(GT_GRID, dtype=np.uint8) * LBL_FREE
        occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = LBL_MANMADE
        
        # 3D Box 體積填充
        if len(curr_boxes['names']) > 0:
            occ, filled = fill_box_interior(occ, curr_boxes['boxes_lidar'], curr_boxes['names'], GT_BOUNDS, GT_VOXEL, GT_GRID)
            total_filled += filled
        
        save_path = os.path.join(out_dir, frame_id)
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(os.path.join(save_path, 'labels.npz'), semantics=occ)
    
    print(f"\n[RAW Pipeline Complete] 總共有 {total_filled} 個動態物件 Voxel 被 3D Box 霸道填滿！")

def generate_occupancy_heuristic(dataroot, pose_dict, num_sweeps=40, out_dir='output'):
    """360° 原始 LiDAR 堆疊 + 高度啟發式判斷道路 vs 建築，動態物件用 3D Box 填充"""
    pcd_files = sorted(glob.glob(os.path.join(dataroot, 'VLS128_pcdnpy', '*.pcd')))
    print(f"[INFO] HEURISTIC 模式：360° LiDAR 堆疊 + 高度判斷道路 + 3D Box 填充")
    
    box_data = load_pkl(os.path.join(dataroot, '3dbox_result.pkl'))
    box_dict = {}
    for item in box_data:
        frame_id = item['file_name'].replace('.pcd', '')
        valid = item['score'] >= 0.4
        box_dict[frame_id] = {
            'names': item['name'][valid],
            'boxes_lidar': item['boxes_lidar'][valid],
            'pred_labels': item['pred_labels'][valid]
        }
    
    frames = [os.path.splitext(os.path.basename(f))[0] for f in pcd_files]
    total_filled = 0
    os.makedirs(out_dir, exist_ok=True)
    
    for i, frame_id in enumerate(tqdm(frames, desc='Generating HEURISTIC Occupancy')):
        curr_pose = pose_dict[frame_id]['matrix']
        curr_pose_inv = np.linalg.inv(curr_pose)
        curr_boxes = box_dict.get(frame_id, {'names': [], 'boxes_lidar': []})
        
        min_b = np.array(GT_BOUNDS[:3])
        max_b = np.array(GT_BOUNDS[3:])
        
        pcd = o3d.io.read_point_cloud(pcd_files[i])
        curr_pts = np.asarray(pcd.points)
        pcs = [curr_pts]
        
        for j in range(1, num_sweeps + 1):
            if i - j < 0: break
            prev_id = frames[i - j]
            prev_pose = pose_dict[prev_id]['matrix']
            prev_pcd = o3d.io.read_point_cloud(pcd_files[i - j])
            prev_pts = np.asarray(prev_pcd.points)
            prev_boxes = box_dict.get(prev_id, {'boxes_lidar': []})['boxes_lidar']
            if len(prev_boxes) > 0:
                mask = points_in_boxes(prev_pts, prev_boxes)
                prev_pts = prev_pts[~mask]
            if len(prev_pts) > 0:
                p_homo = np.hstack((prev_pts, np.ones((len(prev_pts), 1))))
                transformed = (curr_pose_inv @ prev_pose @ p_homo.T).T[:, :3]
                pcs.append(transformed)
        
        for j in range(1, num_sweeps + 1):
            if i + j >= len(frames): break
            next_id = frames[i + j]
            next_pose = pose_dict[next_id]['matrix']
            next_pcd = o3d.io.read_point_cloud(pcd_files[i + j])
            next_pts = np.asarray(next_pcd.points)
            next_boxes = box_dict.get(next_id, {'boxes_lidar': []})['boxes_lidar']
            if len(next_boxes) > 0:
                mask = points_in_boxes(next_pts, next_boxes)
                next_pts = next_pts[~mask]
            if len(next_pts) > 0:
                p_homo = np.hstack((next_pts, np.ones((len(next_pts), 1))))
                transformed = (curr_pose_inv @ next_pose @ p_homo.T).T[:, :3]
                pcs.append(transformed)
        
        all_pts = np.vstack(pcs)
        
        # 體素化 + 高度啟發式標籤
        mask = np.all((all_pts >= min_b) & (all_pts < max_b), axis=1)
        valid_xyz = all_pts[mask]
        idxs = np.clip(((valid_xyz - min_b) / GT_VOXEL).astype(int), 0, np.array(GT_GRID) - 1)
        occ = np.ones(GT_GRID, dtype=np.uint8) * LBL_FREE
        
        # ELAN LiDAR 座標系：地面 z ≈ -2m, LiDAR 在車頂 z ≈ 0
        z_real = valid_xyz[:, 2]
        lbl = np.full(len(z_real), LBL_MANMADE, dtype=np.uint8)
        lbl[z_real < -1.5] = LBL_ROAD  # 地面以下 → 道路
        occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = lbl
        
        # 3D Box 體積填充
        if len(curr_boxes['names']) > 0:
            occ, filled = fill_box_interior(occ, curr_boxes['boxes_lidar'], curr_boxes['names'], GT_BOUNDS, GT_VOXEL, GT_GRID)
            total_filled += filled
        
        save_path = os.path.join(out_dir, frame_id)
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(os.path.join(save_path, 'labels.npz'), semantics=occ)
    
    print(f"\n[HEURISTIC Pipeline Complete] 總共有 {total_filled} 個動態物件 Voxel 被 3D Box 霸道填滿！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='kiss_slam_elan')
    parser.add_argument('--scene', default='citystreet_sunny_day_2025-09-25-15-38-56')
    parser.add_argument('--sweeps', type=int, default=40)
    parser.add_argument('--mode', default='all', choices=['semantic', 'raw', 'heuristic', 'all'],
                        help='semantic=語意點雲, raw=純原始, heuristic=高度判斷, all=三種都跑')
    parser.add_argument('--data_root', default='/home/t113c52027/t113c52027/occ_gt_v2/data/elan')
    parser.add_argument('--out_root', default=os.path.join(os.path.dirname(__file__), 'output'))
    args = parser.parse_args()

    scene_path = os.path.join(args.data_root, args.scene)
    import importlib
    backend_mod = importlib.import_module(f'pose_backends.{args.backend}')
    pose_dict = backend_mod.get_pose_dict(scene_path)
    
    # 輸出結構：output/<scene>/seg/ raw/ heuristic/
    scene_out = os.path.join(args.out_root, args.scene)
    
    if args.mode in ('semantic', 'all'):
        out_dir = os.path.join(scene_out, 'seg')
        generate_occupancy(scene_path, pose_dict, args.sweeps, out_dir=out_dir)
    
    if args.mode in ('raw', 'all'):
        out_dir_raw = os.path.join(scene_out, 'raw')
        generate_occupancy_raw(scene_path, pose_dict, args.sweeps, out_dir=out_dir_raw)
    
    if args.mode in ('heuristic', 'all'):
        out_dir_heur = os.path.join(scene_out, 'heuristic')
        generate_occupancy_heuristic(scene_path, pose_dict, args.sweeps, out_dir=out_dir_heur)

if __name__ == '__main__':
    main()
