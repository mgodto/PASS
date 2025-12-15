import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ==========================================
# 設定 14 維度的對應標籤
# ==========================================
# 順序必須與 extract_subspace_features.py 一致
PARTITION_ORDER = [
    'Full Body', 
    'Head', 
    'Trunk', 
    'Left Arm', 
    'Right Arm', 
    'Left Leg', 
    'Right Leg'
]

# 每個部位對應的顏色 (與之前的視覺化統一)
COLORS = [
    'black',   # Full Body
    '#d62728', # Head (Red)
    '#2ca02c', # Trunk (Green)
    '#1f77b4', # Left Arm (Blue)
    '#17becf', # Right Arm (Cyan)
    '#9467bd', # Left Leg (Purple)
    '#e377c2'  # Right Leg (Pink)
]

def check_feature_file(file_path):
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 {file_path}")
        return

    # 1. 讀取 .npy
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"讀取錯誤：{e}")
        return

    print(f"\n======== 檔案檢查報告 ========")
    print(f"檔案名稱: {os.path.basename(file_path)}")
    print(f"資料形狀 (Frames, Channels): {data.shape}")
    
    # 預期維度檢查
    if data.shape[1] != 14:
        print(f"⚠️ 警告：預期應該有 14 個通道 (7部位 x 2特徵)，但讀取到 {data.shape[1]} 個。")
    
    # NaN / Inf 檢查
    if np.isnan(data).any():
        print(f"⚠️ 警告：資料中包含 NaN (空值)！")
    else:
        print(f"✅ 資料乾淨 (無 NaN)")
        
    if np.isinf(data).any():
        print(f"⚠️ 警告：資料中包含 Inf (無限大)！")

    # 數值範圍檢查 (印出每個部位的最大值，確認是否有訊號)
    print("\n-------- 各部位訊號強度檢查 (Max Value) --------")
    print(f"{'Part Name':<12} | {'1st DS (Vel)':<15} | {'2nd DS (Acc)':<15}")
    print("-" * 48)
    
    has_zero_signal = False
    for i, part_name in enumerate(PARTITION_ORDER):
        idx_1st = i * 2
        idx_2nd = i * 2 + 1
        
        max_1st = np.max(data[:, idx_1st])
        max_2nd = np.max(data[:, idx_2nd])
        
        # 標記全為 0 的異常
        mark = ""
        if max_1st == 0 and max_2nd == 0:
            mark = "⚠️ (No Signal)"
            has_zero_signal = True
            
        print(f"{part_name:<12} | {max_1st:.4f}          | {max_2nd:.4f} {mark}")
    
    if has_zero_signal:
        print("\n⚠️ 注意：部分部位數值全為 0，請確認是否因為該部位點數不足或 SVD 計算失敗。")
    else:
        print("\n✅ 所有部位皆有訊號。")

    # 2. 繪圖驗證
    plot_features(data, os.path.basename(file_path))

def plot_features(data, filename):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 設定時間軸
    frames = np.arange(data.shape[0])
    
    # --- Plot 1: 1st Order DS (Velocity) ---
    ax1 = axes[0]
    ax1.set_title(f"1st Order DS (Velocity) - {filename}")
    ax1.set_ylabel("Dissimilarity")
    ax1.grid(True, alpha=0.3)
    
    for i, part_name in enumerate(PARTITION_ORDER):
        idx = i * 2  # 偶數索引是 1st DS
        ax1.plot(frames, data[:, idx], label=part_name, color=COLORS[i], linewidth=1.5)
    
    ax1.legend(loc='upper right', ncol=2, fontsize='small')
    
    # --- Plot 2: 2nd Order DS (Acceleration) ---
    ax2 = axes[1]
    ax2.set_title("2nd Order DS (Acceleration)")
    ax2.set_ylabel("Dissimilarity")
    ax2.set_xlabel("Frame Index")
    ax2.grid(True, alpha=0.3)
    
    for i, part_name in enumerate(PARTITION_ORDER):
        idx = i * 2 + 1  # 奇數索引是 2nd DS
        ax2.plot(frames, data[:, idx], label=part_name, color=COLORS[i], linewidth=1.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and Visualize 14D Subspace Features")
    parser.add_argument('--input', type=str, required=True, help="Path to the _features.npy file")
    
    args = parser.parse_args()
    check_feature_file(args.input)