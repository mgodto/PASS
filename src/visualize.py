# src/visualize.py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# MediaPipe Poseの33個の関節点の接続情報
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32)
]

def create_skeleton_animation(skeleton_data: np.ndarray):
    """
    単一の骨格データ (.npy) を受け取り、3Dアニメーションとして表示する。
    
    Args:
        skeleton_data (np.ndarray): 形状が (Frames, Landmarks, 3) の骨格データ
    """
    num_frames, num_landmarks, _ = skeleton_data.shape

    # --- 最終修正: アニメーション表示用のデータのコピーを作成 ---
    corrected_data = np.copy(skeleton_data)
    
    # --- 最終修正: 座標を正しく変換 ---
    # MediaPipeの標準は X=左右, Y=上下, Z=奥行き
    # 目的は X=左右, Y=奥行き, Z=上下の標準的なプロットに合わせること
    
    # 1. X軸を反転させる
    corrected_data[:, :, 0] *= -1 
    
    # 2. Y軸とZ軸を入れ替える
    temp_y = np.copy(corrected_data[:, :, 1])
    corrected_data[:, :, 1] = corrected_data[:, :, 2]
    corrected_data[:, :, 2] = temp_y
    
    # 3. 新しいZ軸（元のY軸）の向きを反転させる (これが解決策)
    corrected_data[:, :, 2] *= -1
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_index):
        ax.clear()
        
        # 現在のフレームのデータを取得（修正済みデータから）
        points = corrected_data[frame_index]
        
        # 散布図として関節点をプロット
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='o')

        # 関節点間を線で結ぶ
        for connection in POSE_CONNECTIONS:
            p1_idx, p2_idx = connection
            if p1_idx < num_landmarks and p2_idx < num_landmarks:
                p1 = points[p1_idx]
                p2 = points[p2_idx]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-')

        # 軸の範囲とラベルを設定
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        ax.set_zlim([-1.0, 1.0])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Frame {frame_index}/{num_frames - 1}")

    ani = FuncAnimation(fig, update, frames=num_frames, interval=33)
    plt.show()