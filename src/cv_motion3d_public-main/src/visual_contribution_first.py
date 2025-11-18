# visual_contribution_first.py (簡化後的版本)

import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

# 假設這些是您需要的輔助函式和設定
from config import Confing
from utils import read_c3d, cal_magnitude, gen_shape_subspace, gen_shape_difference_subspace, gram_schmidt
from util.display import display_motion_score_contribution
from util.preprocess import remove_nan

# --- 1. 簡化 argparse：只接收三個必要的完整路徑 ---
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Full path to the input .c3d file.')
parser.add_argument('--output_gif', required=True, help='Full path where the output .gif file should be saved.')
parser.add_argument('--output_npy', required=True, help='Full path where the output .npy file should be saved.')
args = parser.parse_args()

# --- 2. 直接使用從 main.py 傳遞過來的路徑 ---
input_path = args.input
gif_save_path = args.output_gif
npy_save_path = args.output_npy

# --- 3. 核心計算邏輯 (這部分來自您的原始程式碼，保持不變) ---
cfg = Confing()
tau = cfg.interval
data = read_c3d(input_path)
data = remove_nan(data)
num_frame = data.shape[2]

mag_list = []
frame_list = []
f = tau // 2
contribution_list = []

for i in tqdm(range(num_frame - tau * 2)):
    S1 = gen_shape_subspace(data[:, :, i], cfg)
    S2 = gen_shape_subspace(data[:, :, i + tau], cfg)
    D = gen_shape_difference_subspace(S1, S2, cfg)
    P = D @ D.T
    V = P @ S1
    V = V / np.linalg.norm(V, axis=0)
    V = gram_schmidt(V)
    V = np.square(V)
    V = np.sum(V, axis=1)
    mag = cal_magnitude(S1, S2)
    mag_list.append(mag)
    frame_list.append(f)
    f += 1
    contribution_list.append(V)

# --- 4. 使用 main.py 提供的完整路徑直接儲存檔案 ---
print(f"Saving NPY file to: {npy_save_path}")
mag_list_array = np.array(mag_list)
np.save(npy_save_path, mag_list_array)
print(mag_list_array)

print(f"Saving GIF file to: {gif_save_path}")
display_motion_score_contribution(
    input_path,
    frame_list,
    mag_list,
    contribution_list,
    gif_save_path
)

print(f"Successfully processed {os.path.basename(input_path)}")