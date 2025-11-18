# visualize_combined.py

import argparse
import numpy as np
from tqdm import tqdm
import os

# 1. 導入所有需要的輔助函式和設定
from config import Confing
from utils import (
    read_c3d, cal_magnitude, gen_shape_subspace, 
    gen_shape_difference_subspace, gen_shape_principal_com_subspace, 
    gram_schmidt
)
from util.display import display_motion_some_score # <-- ★★★ 關鍵：導入多曲線繪圖函式
from util.preprocess import remove_nan

# --- 2. 設定命令列參數 ---
parser = argparse.ArgumentParser(description="Visualize first and second dissimilarity scores combined.")
parser.add_argument('--input', required=True, help='Full path to the input .c3d file.')
parser.add_argument('--output_gif', required=True, help='Full path where the output .gif file should be saved.')
args = parser.parse_args()

# --- 3. 讀取並預處理 C3D 檔案 (只需執行一次) ---
print(f"Reading data from: {args.input}")
cfg = Confing()
tau = cfg.interval
data = read_c3d(args.input)
data = remove_nan(data)
num_frame = data.shape[2]

# --- 4. 計算 First Dissimilarity Score ---
print("Calculating First Dissimilarity Score...")
mag_list_first = []
frame_list = []
f_first = tau // 2

for i in tqdm(range(num_frame - tau * 2), desc="First DS"):
    S1 = gen_shape_subspace(data[:, :, i], cfg)
    S2 = gen_shape_subspace(data[:, :, i + tau], cfg)
    mag = cal_magnitude(S1, S2)
    mag_list_first.append(mag)
    frame_list.append(f_first)
    f_first += 1

# --- 5. 計算 Second Dissimilarity Score ---
print("Calculating Second Dissimilarity Score...")
mag_list_second = []
f_second = tau * 2 // 2

for i in tqdm(range(num_frame - tau * 2), desc="Second DS"):
    S1 = gen_shape_subspace(data[:, :, i], cfg)
    S2 = gen_shape_subspace(data[:, :, i + tau], cfg)
    S3 = gen_shape_subspace(data[:, :, i + tau * 2], cfg)
    M = gen_shape_principal_com_subspace(S1, S3, cfg)
    mag = cal_magnitude(S2, M)
    mag_list_second.append(mag)
    # frame_list 已經在上面生成，這裡不需要再處理

# --- 6. 準備傳遞給繪圖函式的資料 ---
# 將兩個 list 組合成一個 numpy array
combined_scores = np.array([mag_list_first, mag_list_second])
# 定義圖例標籤
score_labels = ["first DS", "second DS"]

# --- 7. 呼叫多曲線繪圖函式來生成並儲存 GIF ---
print(f"Generating combined GIF, saving to: {args.output_gif}")
display_motion_some_score(
    path=args.input,
    x=frame_list,
    y=combined_scores,
    y_label=score_labels,
    save_path=args.output_gif
)

print(f"Successfully created combined visualization for {os.path.basename(args.input)}")