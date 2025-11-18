# file: check_data.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 這是 MediaPipe 33 個關節點的標準骨骼連接關係
# 我們直接從 stgcn_models.py 複製過來，讓這個腳本可以獨立執行
EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], 
    [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], 
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [11, 23], [12, 24], 
    [23, 24], [23, 25], [25, 27], [27, 29], [27, 31], [24, 26], [26, 28], 
    [28, 30], [28, 32]
]

def visualize_skeleton(file_path):
    """
    讀取一個 .npy 檔案，並將其中一幀的 3D 骨架可視化。
    """
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 '{file_path}'")
        return

    # 1. 載入數據
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return

    # 2. 印出數據形狀
    print(f"成功讀取檔案: '{os.path.basename(file_path)}'")
    print(f"數據形狀 (Frames, Joints, Coords): {data.shape}")

    if data.ndim != 3 or data.shape[1] != 33 or data.shape[2] != 3:
        print("警告：數據形狀不符合預期的 (T, 33, 3) 格式，可能無法正確繪圖。")
        return
        
    # 3. 選擇中間的一幀進行可視化
    frame_index = data.shape[0] // 2
    skeleton_frame = data[frame_index]
    print(f"正在可視化第 {frame_index} 幀...")

    # 4. 建立 3D 繪圖
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 繪製關節點
    x = skeleton_frame[:, 0]
    y = skeleton_frame[:, 1]
    z = skeleton_frame[:, 2]
    ax.scatter(x, y, z, c='red', marker='o')

    # 繪製骨骼連接
    for edge in EDGES:
        p1_idx, p2_idx = edge
        p1 = skeleton_frame[p1_idx]
        p2 = skeleton_frame[p2_idx]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-')

    # 設定圖表
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Skeleton Visualization for Frame {frame_index}')
    ax.view_init(elev=20, azim=-70) # 調整視角
    
    # 確保座標軸比例一致
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 5. 儲存圖片
    save_path = 'skeleton_visualization.png'
    plt.savefig(save_path)
    print(f"\n可視化結果已儲存至 '{save_path}'")
    print("請打開該圖片，檢查骨架是否看起來正常。")
    plt.close()


if __name__ == '__main__':
    # --- ★★★ 請修改這裡 ★★★ ---
    # 請將下面的路徑替換成您想要檢查的任意一個 .npy 檔案的完整路徑。
    # 您可以從您 STGCN_PATHS_PATH 對應的那個 .npy 檔案中隨便複製一個路徑過來。
    file_to_check = "data/processed_skeletons/20160120_ASD_lat__V1-0001.npy" 
    # 例如: file_to_check = "data/stgcn_npy/asd_subject_01_video_01.npy"
    # --- ★★★ 修改結束 ★★★ ---
    
    visualize_skeleton(file_to_check)