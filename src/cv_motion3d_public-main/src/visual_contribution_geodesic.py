import sys
import argparse
from ezc3d import c3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from config import Confing
from utils import read_c3d,gen_shape_subspace,cal_magnitude,gen_shape_difference_subspace
from utils import gen_shape_principal_com_subspace,gram_schmidt
from utils import along_geodesic, orth_decomposition_geodesic
from util.display import display_motion_some_score
from util.preprocess import remove_nan
import numpy as np
from tqdm import tqdm
import os

# コマンドライン引数からパスを取得
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

cfg = Confing()
tau = cfg.interval
data = read_c3d(input_path)
data = remove_nan(data)
num_frame = data.shape[2]


mag1_list = []
mag2_list = []
frame_list = []
f = tau*2 // 2

contribution_list = []

for i in tqdm(range(num_frame-tau*2)):

    if np.isnan(data[:,:,i]).any() or np.isinf(data[:,:,i]).any():
        print("A contains NaN or inf values")

    S1 = gen_shape_subspace(data[:,:,i],cfg)
    S2 = gen_shape_subspace(data[:,:,i+tau],cfg)
    S3 = gen_shape_subspace(data[:,:,i+tau*2],cfg)

    mag1 = along_geodesic(S1,S2,S3,cfg)    
    mag2 = orth_decomposition_geodesic(S1,S2,S3,cfg)

    mag1_list.append(mag1)
    mag2_list.append(mag2)

    frame_list.append(f)
    f += 1

mag_list = [mag1_list, mag2_list]
mag_list_label = ["along geodesic", "orth to geodesic"]

print(f"Saving NPY file to: {npy_save_path}")
mag_list_array = np.array(mag_list)
np.save(npy_save_path, mag_list_array)
print(mag_list_array)

print(f"Saving GIF file to: {gif_save_path}")

display_motion_some_score(input_path, frame_list, mag_list, mag_list_label, gif_save_path)