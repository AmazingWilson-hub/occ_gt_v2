import cv2
import os
import glob
from tqdm import tqdm
import numpy as np
import argparse

# ELAN 風格色表 (特別為 cv2 轉成 BGR 格式)
OCC3D_COLORS_BGR = np.array([
    [0, 0, 0],             # 0:  noise → black
    [144, 128, 112],       # 1:  barrier → gray
    [235, 191,  43],       # 2:  bicycle → 天藍
    [ 33,  91, 255],       # 3:  bus → 橘
    [ 60,  20, 220],       # 4:  car → 紅
    [ 33,  91, 255],       # 5:  construction_vehicle → 橘
    [  0,  61, 135],       # 6:  motorcycle → 深棕
    [ 75,   0,  76],       # 7:  pedestrian → 深紫
    [144, 128, 112],       # 8:  traffic_cone → gray
    [ 33,  91, 255],       # 9:  trailer → 橘
    [ 33,  91, 255],       # 10: truck → 橘
    [248, 150, 215],       # 11: driveable_surface → 紫粉 (road)
    [ 70, 206, 247],       # 12: other_flat → 黃
    [ 70, 206, 247],       # 13: sidewalk → 黃
    [152, 251, 152],       # 14: terrain → 淡綠
    [ 70, 206, 247],       # 15: manmade → 黃
    [152, 251, 152],       # 16: vegetation → 綠
    [255, 255, 255],       # 17: free_space → 白
], dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Generate layout video with 6 cameras and clean BEV")
    parser.add_argument('--scene', default='test_2026-03-23-10-42-37', help='Scene name')
    parser.add_argument('--data_root', default='/home/t113c52027/t113c52027/occ_gt_v2/data/u5', help='Root of input data')
    parser.add_argument('--bev_root', default='/home/t113c52027/t113c52027/occ_gt_v2/cvpr_format_occ_gen_u5/output', help='Root of output BEV images')
    parser.add_argument('--out_dir', default='/home/t113c52027/t113c52027/occ_gt_v2/cvpr_format_occ_gen_u5/video_gen', help='Output directory for videos')
    # 注意：這裡直接讀取 numpy 陣列資料即可 (heuristic)，不需要先轉好的 vis_heuristic
    parser.add_argument('--mode', default='heuristic', help='Subdirectory inside scene output (e.g. heuristic, raw)')
    parser.add_argument('--fps', type=int, default=10, help='Video FPS')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.scene}_{args.mode}_with_ego.mp4')

    scene_dir = os.path.join(args.data_root, args.scene)
    pred_dir = os.path.join(args.bev_root, args.scene, args.mode)

    # 支援新舊命名格式，有些場景已經被改名為 cam_xxx，有些還是 port_xxx
    cam_top = [
        (['cam_front_left', 'port_2_camera'], 'Front Left'),
        (['cam_front', 'port_8_camera'], 'Front'),
        (['cam_front_right', 'port_5_camera'], 'Front Right')
    ]
    cam_bottom = [
        (['cam_back_left', 'port_3_camera'], 'Back Left'),
        (['cam_back', 'port_7_camera'], 'Back'),
        (['cam_back_right', 'port_6_camera'], 'Back Right')
    ]

    # 直接掃描有產出 labels.npz 的 Frame
    frames = sorted([d for d in os.listdir(pred_dir) if os.path.exists(os.path.join(pred_dir, d, 'labels.npz'))])
    if not frames:
        print(f"ERROR: No labels.npz found in {pred_dir}")
        return

    print(f"Found {len(frames)} frames to process for scene {args.scene}.")

    cam_w, cam_h = 640, 426
    bev_s = cam_h * 2  

    total_w = cam_w * 3 + bev_s
    total_h = cam_h * 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, args.fps, (total_w, total_h))

    for frame in tqdm(frames, desc="Generating Video"):
        frame_canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # 繪製上排相機
        for idx, (dir_candidates, cam_name) in enumerate(cam_top):
            img_path = None
            for cand in dir_candidates:
                p = os.path.join(scene_dir, cand, f"{frame}.jpg")
                if os.path.exists(p):
                    img_path = p
                    break
            
            if img_path is not None:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (cam_w, cam_h))
                    cv2.putText(img, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    frame_canvas[0:cam_h, idx*cam_w:(idx+1)*cam_w] = img

        # 繪製下排相機
        for idx, (dir_candidates, cam_name) in enumerate(cam_bottom):
            img_path = None
            for cand in dir_candidates:
                p = os.path.join(scene_dir, cand, f"{frame}.jpg")
                if os.path.exists(p):
                    img_path = p
                    break
            
            if img_path is not None:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (cam_w, cam_h))
                    cv2.putText(img, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    frame_canvas[cam_h:2*cam_h, idx*cam_w:(idx+1)*cam_w] = img

        # 即時由 NPZ 渲染 BEV 圖像
        npz_path = os.path.join(pred_dir, frame, 'labels.npz')
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            grid = data['semantics'] # 形狀為 (200, 200, 20)
            
            # 從上到下壓扁
            nx, ny, nz = grid.shape
            bev_map = np.ones((nx, ny), dtype=np.uint8) * 17
            for z in range(nz):
                slice_z = grid[:, :, z]
                mask = slice_z != 17
                bev_map[mask] = slice_z[mask]
                
            # 套用色彩 (直接填入 BGR 陣列)
            bev_bgr = np.zeros((nx, ny, 3), dtype=np.uint8)
            for label in np.unique(bev_map):
                if 0 <= label <= 17:
                    bev_bgr[bev_map == label] = OCC3D_COLORS_BGR[int(label)]
            
            # 轉換座標系：
            # numpy 的 x (0~199) 在這包資料是從車後到車前(-40~40)
            # numpy 的 y (0~199) 是由右到左(-40~40)
            # 翻轉座標，讓視覺變成：前方位在最上面(Top)，左方在最左邊(Left)
            # 加入 np.ascontiguousarray 修正 cv2 報錯
            bev_img = np.ascontiguousarray(bev_bgr[::-1, ::-1, :])
            
            # 畫上精準本車！網格中心座標為 `[100, 100]`
            # 一格 voxel = 0.4m
            # 車寬約 2m = 6 pixels, 車長約 4.8m = 12 pixels
            cx, cy = 100, 100
            # 畫「偏綠的青綠底白框」代表本車 (BGR 格式: 減少藍色，增加綠色比例，如 50, 255, 0)
            cv2.rectangle(bev_img, (cx - 3, cy - 6), (cx + 3, cy + 6), (50, 255, 0), -1)  # 偏綠的青綠色車體
            cv2.rectangle(bev_img, (cx - 3, cy - 6), (cx + 3, cy + 6), (255, 255, 255), 1)   # 白色線條勾勒

            # 放大到佔滿右側版面 (使用最近鄰插值 INTER_NEAREST 讓體素網格邊緣維持銳利不模糊)
            bev_large = cv2.resize(bev_img, (bev_s, bev_s), interpolation=cv2.INTER_NEAREST)
            
            cv2.putText(bev_large, f"BEV ({args.mode})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            frame_canvas[0:bev_s, 3*cam_w:3*cam_w+bev_s] = bev_large

        # 在畫面左下角標示 Frame ID
        cv2.putText(frame_canvas, f"Frame: {frame}", (10, total_h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame_canvas)

    out.release()
    print(f"Done! Video saved to {out_path}")

if __name__ == '__main__':
    main()
