import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import os
import argparse
import sys
import cv2

# ==========================================
# 0. 路徑修正
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src', 'cv_motion3d_public-main', 'src')
util_path = os.path.join(src_path, 'util')
sys.path.append(src_path)
sys.path.append(util_path)

from config import Confing
from utils import gen_shape_subspace, cal_magnitude, gen_shape_principal_com_subspace

# ==========================================
# 1. 部位分割與「統一色票」設定
# ==========================================
PARTITIONS = {
    'Full Body': {'indices': list(range(33)), 'dim': 3, 'color': 'black'},
    'Head': {'indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'dim': 3, 'color': '#d62728'}, # 紅
    'Trunk': {'indices': [11, 12, 23, 24], 'dim': 2, 'color': '#2ca02c'}, # 綠
    'Left Arm': {'indices': [13, 15, 17, 19, 21], 'dim': 2, 'color': '#1f77b4'}, # 藍
    'Right Arm': {'indices': [14, 16, 18, 20, 22], 'dim': 2, 'color': '#17becf'}, # 青
    'Left Leg': {'indices': [25, 27, 29, 31], 'dim': 2, 'color': '#9467bd'}, # 紫
    'Right Leg': {'indices': [26, 28, 30, 32], 'dim': 2, 'color': '#e377c2'}  # 粉
}

# 計算用的索引 (需包含根關節以確保維度)
CALC_INDICES = {
    'Full Body': list(range(33)),
    'Head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Trunk': [11, 12, 23, 24],
    'Left Arm': [11, 13, 15, 17, 19, 21],
    'Right Arm': [12, 14, 16, 18, 20, 22],
    'Left Leg': [23, 25, 27, 29, 31],
    'Right Leg': [24, 26, 28, 30, 32]
}

# ==========================================
# 2. 座標轉換函式
# ==========================================
def correct_coordinates(data):
    print("Applying coordinate correction...")
    new_data = np.copy(data)
    original_x = data[:, :, 0]
    original_y = data[:, :, 1]
    original_z = data[:, :, 2]
    new_data[:, :, 0] = original_x * -1
    new_data[:, :, 1] = original_z
    new_data[:, :, 2] = original_y * -1
    return new_data

# ==========================================
# 3. 視覺化函式
# ==========================================
def display_partition_full(data_npy, frames, results_1st, results_2nd, video_path, data_filename, save_path):
    # 轉換維度: (3, 33, Frames)
    point_data = data_npy.transpose(2, 1, 0) 
    
    # 準備顏色陣列
    point_colors = ['#d3d3d3'] * 33 
    for part_name, info in PARTITIONS.items():
        if part_name == 'Full Body': continue
        color = info['color']
        for idx in info['indices']:
            if idx < 33: point_colors[idx] = color
    # 強制設定核心點為綠色
    for idx in [11, 12, 23, 24]:
        point_colors[idx] = PARTITIONS['Trunk']['color']

    # 準備影片
    cap = None
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
    
    # 設定畫布
    fig = plt.figure(figsize=(22, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.5, 1.5])
    
    # --- Panel 1: Video ---
    ax0 = fig.add_subplot(gs[0])
    ax0.set_title(f"ID: {data_filename}", fontsize=12, fontweight='bold')
    ax0.axis('off')
    video_artist = None
    if cap:
        ret, first_frame = cap.read()
        if ret:
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            video_artist = ax0.imshow(first_frame_rgb)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            video_artist = ax0.imshow(np.zeros((100, 100, 3)))
    else:
        ax0.text(0.5, 0.5, "Video Not Found", ha='center')

    # --- Panel 2: 3D Skeleton ---
    ax1 = fig.add_subplot(gs[1], projection='3d')
    
    # ★★★ 修正點 (Fix) ★★★
    # 取得第 0 幀的數據作為初始值，確保點數 (33) 與顏色數 (33) 一致
    init_pose = point_data[:, :, 0] 
    sc = ax1.scatter(init_pose[0], init_pose[1], init_pose[2], 
                     s=30, c=point_colors, edgecolors='black', alpha=1.0)
    
    connections = [[11,12], [11,23], [12,24], [23,24], 
                   [11,13], [13,15], [12,14], [14,16], 
                   [23,25], [25,27], [24,26], [26,28]]
    lines_3d = [ax1.plot([],[],[], 'black', alpha=0.3)[0] for _ in connections]

    title_text = ax1.set_title(f"Frame: {0}")
    
    # 設定軸範圍
    x_min, x_max = np.nanmin(point_data[0]), np.nanmax(point_data[0])
    y_min, y_max = np.nanmin(point_data[1]), np.nanmax(point_data[1])
    z_min, z_max = np.nanmin(point_data[2]), np.nanmax(point_data[2])
    ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max); ax1.set_zlim(z_min, z_max)
    ax1.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
    ax1.view_init(elev=10, azim=20)

    # --- Panel 3 & 4: Curves ---
    ax2 = fig.add_subplot(gs[2])
    ax2.set_title("1st Order DS (Velocity)")
    ax2.set_xlabel("Frame"); ax2.set_ylabel("Dissimilarity")
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[3])
    ax3.set_title("2nd Order DS (Acceleration)")
    ax3.set_xlabel("Frame")
    ax3.grid(True, alpha=0.3)

    lines_1st, lines_2nd = {}, {}
    
    # 設定 Y 軸最大值
    all_scores_1 = [s for sublist in results_1st.values() for s in sublist]
    max_y1 = max(all_scores_1) * 1.1 if all_scores_1 else 1.0
    all_scores_2 = [s for sublist in results_2nd.values() for s in sublist]
    max_y2 = max(all_scores_2) * 1.1 if all_scores_2 else 1.0
    
    ax2.set_xlim(frames[0], frames[-1]); ax2.set_ylim(0, max_y1)
    ax3.set_xlim(frames[0], frames[-1]); ax3.set_ylim(0, max_y2)

    for part_name in results_1st.keys():
        color = PARTITIONS[part_name]['color']
        l1, = ax2.plot([], [], label=part_name, color=color, linewidth=1.5)
        lines_1st[part_name] = l1
        l2, = ax3.plot([], [], label=part_name, color=color, linewidth=1.5)
        lines_2nd[part_name] = l2
        
    ax2.legend(loc='upper right', ncol=2, fontsize='x-small')

    # --- Update Function ---
    def update(frame_idx):
        artists = []
        
        # 1. Video
        if cap and video_artist:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) 
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_artist.set_data(frame_rgb)
                artists.append(video_artist)

        # 2. Skeleton
        current_pose = point_data[:, :, frame_idx]
        sc._offsets3d = (current_pose[0], current_pose[1], current_pose[2])
        artists.append(sc)
        
        for line, (i, j) in zip(lines_3d, connections):
            line.set_data(current_pose[0, [i, j]], current_pose[1, [i, j]])
            line.set_3d_properties(current_pose[2, [i, j]])
            artists.extend(lines_3d)

        title_text.set_text(f"Frame: {frame_idx}")
        
        # 3. Curves
        if frames[0] <= frame_idx <= frames[-1]:
            curr_len = frame_idx - frames[0] + 1
            for part_name, score_list in results_1st.items():
                lines_1st[part_name].set_data(frames[:curr_len], score_list[:curr_len])
                artists.append(lines_1st[part_name])
            for part_name, score_list in results_2nd.items():
                lines_2nd[part_name].set_data(frames[:curr_len], score_list[:curr_len])
                artists.append(lines_2nd[part_name])
                
        return artists

    print(f"Generating animation to {save_path} ...")
    valid_frames = range(frames[0], frames[-1] + 1)
    ani = FuncAnimation(fig, update, frames=valid_frames, interval=50, blit=False)
    ani.save(save_path, writer='pillow', fps=20)
    print("Done.")
    if cap: cap.release()

# ==========================================
# 4. 主分析流程
# ==========================================
def analyze_partitions(npy_path, video_path, output_path):
    cfg = Confing()
    tau = cfg.interval
    
    if not os.path.exists(npy_path):
        print(f"Error: File not found {npy_path}")
        return

    full_data = np.load(npy_path)
    full_data = correct_coordinates(full_data)
    num_frames = full_data.shape[0]
    
    frame_indices = list(range(0, num_frames - 2 * tau))
    results_1st = {}
    results_2nd = {}
    original_cfg_dim = cfg.subspace_dim
    base_filename = os.path.basename(npy_path).split('.')[0]
    
    print("Calculating features...")

    for part_name, info in PARTITIONS.items():
        calc_indices = CALC_INDICES[part_name]
        target_dim = info['dim']
        cfg.subspace_dim = target_dim
        
        scores_1st = []
        scores_2nd = []
        
        for i in tqdm(frame_indices, desc=part_name, leave=False):
            p1 = full_data[i, calc_indices, :].T
            p2 = full_data[i + tau, calc_indices, :].T
            p3 = full_data[i + 2*tau, calc_indices, :].T
            
            try:
                S1 = gen_shape_subspace(p1, cfg)
                S2 = gen_shape_subspace(p2, cfg)
                S3 = gen_shape_subspace(p3, cfg)
                
                mag1 = cal_magnitude(S1, S2)
                M = gen_shape_principal_com_subspace(S1, S3, cfg)
                mag2 = cal_magnitude(S2, M)
            except:
                mag1 = 0.0; mag2 = 0.0
                
            scores_1st.append(mag1)
            scores_2nd.append(mag2)
            
        results_1st[part_name] = scores_1st
        results_2nd[part_name] = scores_2nd
        
    cfg.subspace_dim = original_cfg_dim

    if video_path is None:
        potential_video_dir = os.path.join(current_dir, 'dataset') 
        potential_path = os.path.join(potential_video_dir, base_filename + ".mp4")
        if os.path.exists(potential_path):
            video_path = potential_path
            print(f"Auto-detected video: {video_path}")

    display_partition_full(full_data, frame_indices, results_1st, results_2nd, video_path, base_filename, output_path)

if __name__ == "__main__":
    default_input = "data/processed_skeletons/20160120_ASD_lat__V1-0001.npy"
    default_video = "data/segmentation_dataset_512/alldata/train/20160120_ASD_lat__V1-0001.mp4"
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=default_input)
    parser.add_argument('--video', type=str, default=default_video)
    parser.add_argument('--output', type=str, default="combined_colored_analysis.gif")
    args = parser.parse_args()
    
    analyze_partitions(args.input, args.video, args.output)