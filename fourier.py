import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import argparse
import os

try:
    from src.config import SUBSPACE_NPY_DIR
except ImportError:
    SUBSPACE_NPY_DIR = 'results/subspace_npy'

def fourier_analysis(filepath, sampling_rate=30):
    """
    對給定的.npy時間序列檔案進行傅立葉分析，並視覺化結果。

    Args:
        filepath (str): .npy 檔案的路徑。
        sampling_rate (int): 原始影片的採樣率 (幀/秒)。預設為30。
    """
    filepath = os.path.join(SUBSPACE_NPY_DIR, filepath) if not os.path.isabs(filepath) else filepath

    if not os.path.exists(filepath):
        print(f"錯誤：找不到檔案 '{filepath}'")
        return

    print(f"--- 正在分析檔案: {os.path.basename(filepath)} ---")
    
    # 1. 載入數據
    waveform = np.load(filepath)
    if waveform.ndim != 1:
        print(f"錯誤：輸入檔案必須是一維陣列，但其形狀為 {waveform.shape}")
        return
        
    num_samples = len(waveform)
    time = np.arange(num_samples) / sampling_rate

    # 2. 執行傅立葉轉換 (FFT)
    # 使用 rfft 是因為我們的輸入是實數，可以提高效率
    fft_vals = np.abs(np.fft.rfft(waveform))
    fft_freqs = np.fft.rfftfreq(num_samples, 1 / sampling_rate)

    # 3. 從頻譜中提取量化特徵
    # a. 主頻率 (Dominant Frequency)
    # 我們忽略直流分量 (第一個元素)，找到能量最高的頻率
    dominant_freq_index = np.argmax(fft_vals[1:]) + 1
    dominant_freq = fft_freqs[dominant_freq_index]
    
    # b. 頻譜熵 (Spectral Entropy)
    # 計算功率譜並正規化為機率分佈
    #print(fft_vals)
    power_spectrum = fft_vals**2
    prob_spectrum = power_spectrum / np.sum(power_spectrum)
    spectral_entropy = entropy(prob_spectrum)

    print("\n--- 分析結果 ---")
    print(f"主頻率 (步頻): {dominant_freq:.4f} Hz")
    print(f"頻譜熵 (穩定性指標): {spectral_entropy:.4f}")

    # 4. 視覺化結果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Fourier Analysis of {os.path.basename(filepath)}', fontsize=16)

    # 上方的圖：時域波形
    ax1.plot(time, waveform)
    ax1.set_title('Time Domain: Dissimilarity Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Dissimilarity Score')
    ax1.grid(True)

    # 下方的圖：頻域頻譜
    
    ax2.plot(fft_freqs, fft_vals)
    ax2.set_title(f'Frequency Domain: Spectrum (Entropy: {spectral_entropy:.4f})')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.axvline(dominant_freq, color='r', linestyle='--', label=f'Dominant Freq: {dominant_freq:.2f} Hz')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 儲存圖片
    os.makedirs('results/figures/fft_analysis', exist_ok=True)
    save_path = f'results/figures/fft_analysis/{os.path.splitext(os.path.basename(filepath))[0]}_fft.png'
    plt.savefig(save_path)
    print(f"分析圖表已儲存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform Fourier analysis on a subspace feature .npy file.')
    parser.add_argument('filepath', type=str, help='Path to the input .npy file from subspace method.')
    parser.add_argument('--fs', type=int, default=30, help='Sampling rate of the original video in Hz (frames per second). Default is 30.')
    args = parser.parse_args()
    
    fourier_analysis(args.filepath, args.fs)