import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from tqdm import tqdm

# 嘗試從src模組導入設定，如果失敗則使用預設路徑
# 這讓腳本可以更獨立地執行
try:
    from src.config import SUBSPACE_NPY_DIR
except ImportError:
    SUBSPACE_NPY_DIR = 'results/subspace_npy'

FIGURES_DIR = 'results/figures'

def get_label_from_filename(filename: str) -> str:
    """
    從給定的檔名中，識別並回傳疾病標籤。
    """
    # 正常樣本的ID不包含疾病標籤，特別處理
    if 'Normal_Control' in filename:
        return 'Normal_Control'
        
    possible_labels = ["ASD", "LCS", "DHS", "HipOA"]
    for label in possible_labels:
        if label in filename:
            return label
    return "Unknown"

def normalize_gait_cycles(waveform, num_points=100):
    """
    將一個時間序列波形分割成獨立的步態週期，並將它們的長度正規化。
    """
    # 根據波形的平均高度和最小間距來尋找波峰，這些參數可以根據數據調整
    peaks, _ = find_peaks(waveform, height=np.mean(waveform) * 0.5, distance=20)
    normalized_cycles = []
    
    # 至少需要兩個波峰才能構成一個週期
    if len(peaks) < 2:
        return normalized_cycles # 回傳空列表
        
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i+1]
        cycle = waveform[start:end]
        
        # 避免週期過短導致的插值錯誤
        if len(cycle) < 2:
            continue

        # 使用線性插值將週期長度正規化為 num_points 個點
        x_original = np.linspace(0, 1, num=len(cycle))
        x_normalized = np.linspace(0, 1, num=num_points)
        f = interp1d(x_original, cycle, kind='linear', fill_value="extrapolate")
        normalized_cycle = f(x_normalized)
        normalized_cycles.append(normalized_cycle)
        
    return normalized_cycles

def plot_averaged_waveforms(feature_type='first', normal_control_id=None):
    """
    繪製每個疾病類別的平均步態波形，並可選地疊加正常對照組的波形。
    """
    print(f"--- Generating Averaged Waveform Plot for '{feature_type}' feature ---")
    
    npy_files = glob.glob(os.path.join(SUBSPACE_NPY_DIR, f"*_{feature_type}_mag.npy"))
    if not npy_files:
        print(f"錯誤：在 {SUBSPACE_NPY_DIR} 中找不到任何 '_{feature_type}_mag.npy' 檔案。")
        return

    # 建立一個包含所有檔案路徑和標籤的列表
    file_info = []
    for p in npy_files:
        # 排除正常對照組的檔案，稍後單獨處理
        if normal_control_id and normal_control_id in p:
            continue
        file_info.append({'label': get_label_from_filename(os.path.basename(p)), 'path': p})

    df = pd.DataFrame(file_info)
    labels = sorted([l for l in df['label'].unique() if l != "Unknown" and l != "Normal_Control"])
    
    plt.figure(figsize=(16, 9)) # 使用更寬的圖表尺寸
    
    # 繪製每個疾病類別的平均波形
    for label in labels:
        print(f"Processing class: {label}...")
        class_df = df[df['label'] == label]
        all_normalized_cycles = []
        for npy_path in class_df['path']:
            waveform = np.load(npy_path)
            normalized_cycles = normalize_gait_cycles(waveform)
            all_normalized_cycles.extend(normalized_cycles)
        
        if not all_normalized_cycles:
            print(f"警告：在類別 {label} 中找不到有效的步態週期。")
            continue
            
        all_normalized_cycles = np.array(all_normalized_cycles)
        mean_waveform = np.mean(all_normalized_cycles, axis=0)
        std_waveform = np.std(all_normalized_cycles, axis=0)
        x_axis = np.linspace(0, 100, num=len(mean_waveform)) # X軸改為百分比
        
        line, = plt.plot(x_axis, mean_waveform, label=f'Mean - {label}', linewidth=2)
        plt.fill_between(x_axis, mean_waveform - std_waveform, mean_waveform + std_waveform, alpha=0.2, color=line.get_color())

    # 如果指定了正常對照組，則單獨繪製它
    if normal_control_id:
        normal_path = os.path.join(SUBSPACE_NPY_DIR, f"{normal_control_id}_{feature_type}_mag.npy")
        if os.path.exists(normal_path):
            print(f"Processing Normal Control: {normal_control_id}...")
            waveform = np.load(normal_path)
            normalized_cycles = normalize_gait_cycles(waveform)
            if normalized_cycles:
                # 將正常樣本自己的所有週期平均，使其波形更穩定
                normal_cycle = np.mean(np.array(normalized_cycles), axis=0) 
                x_axis = np.linspace(0, 100, num=len(normal_cycle))
                plt.plot(x_axis, normal_cycle, label='Normal Control', color='black', linestyle='--', linewidth=2.5)
        else:
            print(f"警告：找不到指定的正常對照組檔案： {normal_path}")

    plt.title(f'Averaged Gait Cycle Waveform vs. Normal Control ({feature_type.capitalize()} Feature)', fontsize=16)
    plt.xlabel('Gait Cycle (%)', fontsize=12)
    plt.ylabel('Dissimilarity Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, f'averaged_waveform_{feature_type}_with_control.png')
    plt.savefig(save_path)
    print(f"\n平均波形圖已儲存至： {save_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced analysis and visualization for gait waveforms.")
    subparsers = parser.add_subparsers(dest='action', required=True, help="Available actions")

    # 'waveforms' 指令
    parser_wave = subparsers.add_parser('waveforms', help='Generate averaged waveform plots for each class.')
    parser_wave.add_argument('--feature', type=str, default='first', choices=['first', 'second', 'geodesic'], help='Feature type to analyze.')
    parser_wave.add_argument('--control_id', type=str, default=None, help='Video ID of the normal control subject to plot as a baseline.')
    
    # 'fft' 指令 (目前為 placeholder)
    parser_fft = subparsers.add_parser('fft', help='Perform frequency analysis and generate new features and plots.')

    args = parser.parse_args()

    if args.action == 'waveforms':
        plot_averaged_waveforms(feature_type=args.feature, normal_control_id=args.control_id)
    elif args.action == 'fft':
        print("FFT analysis is not implemented in this version yet.")

