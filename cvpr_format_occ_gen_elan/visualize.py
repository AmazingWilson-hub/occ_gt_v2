import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ELAN Semantic Color Map (RGB 0-255)
# Indexed by internal Occ3D label ID, using ELAN's original colors
OCC3D_COLORS = np.array([
    [0, 0, 0],             # 0:  noise → black
    [112, 128, 144],       # 1:  barrier → gray
    [ 43, 191, 235],       # 2:  bicycle → ELAN bicycle
    [255,  91,  33],       # 3:  bus → ELAN big car (orange)
    [220,  20,  60],       # 4:  car → ELAN car
    [255,  91,  33],       # 5:  construction_vehicle → ELAN big car
    [135,  61,   0],       # 6:  motorcycle → ELAN motorcycle
    [ 76,   0,  75],       # 7:  pedestrian → ELAN pedestrian
    [112, 128, 144],       # 8:  traffic_cone → gray
    [255,  91,  33],       # 9:  trailer → ELAN big car
    [255,  91,  33],       # 10: truck → ELAN big car
    [215, 150, 248],       # 11: driveable_surface → ELAN road (紫粉)
    [247, 206,  70],       # 12: other_flat → ELAN undriveable area (黃)
    [247, 206,  70],       # 13: sidewalk → ELAN undriveable area (黃)
    [152, 251, 152],       # 14: terrain → green
    [247, 206,  70],       # 15: manmade → ELAN undriveable area (黃)
    [152, 251, 152],       # 16: vegetation → green
    [255, 255, 255],       # 17: free_space → white
])

def visualize_bev(npz_path, output_path):
    data = np.load(npz_path)
    grid = data['semantics']
    
    nx, ny, nz = grid.shape
    bev_map = np.ones((nx, ny), dtype=np.uint8) * 17
    
    for z in range(nz):
        slice_z = grid[:, :, z]
        mask = slice_z != 17
        bev_map[mask] = slice_z[mask]
        
    bev_rgb = np.zeros((nx, ny, 3), dtype=np.uint8)
    
    unique_labels = np.unique(bev_map)
    for label in unique_labels:
        if label < 0 or label > 17:
            continue
        color = OCC3D_COLORS[int(label)]
        bev_rgb[bev_map == label] = color
        
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(bev_rgb, (1, 0, 2)), origin='lower')
    plt.axis('off')
    plt.title(f"BEV Occupancy - ELAN Dataset")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir',
                        default=os.path.join(os.path.dirname(__file__), 'output/citystreet_sunny_day_2025-09-25-15-38-56/seg'),
                        help='e.g. output/citystreet_.../seg')
    parser.add_argument('--out_dir', default=None,
                        help='If not set, auto-derive from pred_dir: .../seg → .../vis_seg')
    args = parser.parse_args()

    # 自動推算 vis 目錄：output/<scene>/seg → output/<scene>/vis_seg
    if args.out_dir is None:
        parent = os.path.dirname(args.pred_dir)  # output/<scene>
        mode_name = os.path.basename(args.pred_dir)  # raw / heuristic / seg
        args.out_dir = os.path.join(parent, f'vis_{mode_name}')

    os.makedirs(args.out_dir, exist_ok=True)
    
    if not os.path.exists(args.pred_dir):
        print(f"ERROR: Cannot find: {args.pred_dir}")
        return

    tokens = [t for t in os.listdir(args.pred_dir) 
              if os.path.isdir(os.path.join(args.pred_dir, t))]
    print(f"Found {len(tokens)} frames to visualize → {args.out_dir}")
    
    for token in sorted(tokens):
        npz_path = os.path.join(args.pred_dir, token, 'labels.npz')
        if not os.path.exists(npz_path):
            continue
        out_img_path = os.path.join(args.out_dir, f"{token}_bev.png")
        visualize_bev(npz_path, out_img_path)
    
    print(f"Done! {len(tokens)} BEV images saved to {args.out_dir}")

if __name__ == '__main__':
    main()
