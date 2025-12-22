import numpy as np
import os
import argparse
import sys
from tqdm import tqdm

# ==========================================
# 0. 路徑修正與設定
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src', 'cv_motion3d_public-main', 'src')
util_path = os.path.join(src_path, 'util')
sys.path.append(src_path)
sys.path.append(util_path)

from config import Confing
from utils import gen_shape_subspace, cal_magnitude, gen_shape_principal_com_subspace

# ★★★ 強制設定輸出路徑 ★★★
from src.config import PARTITION_NPY_DIR

# ★★★ 強制設定輸出路徑 ★★★
FIXED_OUTPUT_DIR = PARTITION_NPY_DIR

# ==========================================
# 1. 定義特徵計算設定
# ==========================================
PARTITION_ORDER = [
    'Full Body', 'Head', 'Trunk', 
    'Left Arm', 'Right Arm', 'Left Leg', 'Right Leg'
]

CALC_INDICES = {
    'Full Body': list(range(33)),
    'Head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Trunk': [11, 12, 23, 24],
    'Left Arm': [11, 13, 15, 17, 19, 21],
    'Right Arm': [12, 14, 16, 18, 20, 22],
    'Left Leg': [23, 25, 27, 29, 31],
    'Right Leg': [24, 26, 28, 30, 32]
}

PARTITION_DIMS = {
    'Full Body': 3, 'Head': 3, 'Trunk': 2,
    'Left Arm': 2, 'Right Arm': 2, 'Left Leg': 2, 'Right Leg': 2
}

# ==========================================
# 2. 核心處理函式
# ==========================================
def correct_coordinates(data):
    """ 座標軸修正 """
    new_data = np.copy(data)
    original_x = data[:, :, 0]
    original_y = data[:, :, 1]
    original_z = data[:, :, 2]
    new_data[:, :, 0] = original_x * -1
    new_data[:, :, 1] = original_z
    new_data[:, :, 2] = original_y * -1
    return new_data

def extract_features_from_file(npy_path, padding=True):
    """
    讀取 .npy 骨架檔，計算 14 維特徵並儲存至固定路徑。
    """
    cfg = Confing()
    tau = cfg.interval
    
    if not os.path.exists(npy_path):
        print(f"Error: File not found {npy_path}")
        return

    # 1. 載入與修正
    try:
        full_data = np.load(npy_path)
    except Exception as e:
        print(f"Failed to load {npy_path}: {e}")
        return

    full_data = correct_coordinates(full_data)
    num_frames = full_data.shape[0]
    
    features_list = []
    # 有效範圍 (因為二階差分需要 t + 2*tau)
    valid_range = range(0, num_frames - 2 * tau)
    original_cfg_dim = cfg.subspace_dim
    
    # 2. 計算特徵
    # 如果幀數太少無法計算，將直接產生空陣列或報錯，這裡做個簡單檢查
    if num_frames <= 2 * tau:
        # 幀數不足，回傳全 0 或跳過
        print(f"Skipping {os.path.basename(npy_path)}: Not enough frames ({num_frames})")
        return None

    for t in valid_range:
        frame_features = []
        for part_name in PARTITION_ORDER:
            indices = CALC_INDICES[part_name]
            target_dim = PARTITION_DIMS[part_name]
            cfg.subspace_dim = target_dim
            
            p1 = full_data[t, indices, :].T
            p2 = full_data[t + tau, indices, :].T
            p3 = full_data[t + 2*tau, indices, :].T
            
            mag1 = 0.0
            mag2 = 0.0
            try:
                S1 = gen_shape_subspace(p1, cfg)
                S2 = gen_shape_subspace(p2, cfg)
                
                # 1st DS
                mag1 = cal_magnitude(S1, S2)
                
                # 2nd DS
                S3 = gen_shape_subspace(p3, cfg)
                M = gen_shape_principal_com_subspace(S1, S3, cfg)
                mag2 = cal_magnitude(S2, M)
            except:
                # 計算失敗 (SVD 不收斂等) 補 0
                mag1 = 0.0
                mag2 = 0.0
            
            frame_features.extend([mag1, mag2])
        features_list.append(frame_features)
    
    cfg.subspace_dim = original_cfg_dim
    features_array = np.array(features_list)
    
    # 3. Padding (補回原始長度)
    if padding and num_frames > features_array.shape[0]:
        pad_width = num_frames - features_array.shape[0]
        # 在後面補 0
        if pad_width > 0:
            padding_arr = np.zeros((pad_width, features_array.shape[1]))
            features_array = np.vstack([features_array, padding_arr])
    
    # 4. 儲存至固定路徑
    if not os.path.exists(FIXED_OUTPUT_DIR):
        os.makedirs(FIXED_OUTPUT_DIR)
        print(f"Created output directory: {FIXED_OUTPUT_DIR}")

    filename = os.path.basename(npy_path)
    save_name = filename.replace('.npy', '_subspace_features.npy')
    output_path = os.path.join(FIXED_OUTPUT_DIR, save_name)
        
    np.save(output_path, features_array)
    return output_path

# ==========================================
# 3. 批次處理邏輯
# ==========================================
def process_batch(input_dir):
    """ 遞迴處理資料夾內所有 .npy 檔 """
    files_to_process = []
    
    print(f"Scanning directory: {input_dir} ...")
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # 條件：是 .npy 且不是已經產生的 feature 檔
            if file.endswith('.npy') and '_subspace_features' not in file:
                files_to_process.append(os.path.join(root, file))
    
    print(f"Found {len(files_to_process)} skeleton files.")
    print(f"Results will be saved to: {FIXED_OUTPUT_DIR}")
    
    count = 0
    for file_path in tqdm(files_to_process):
        res = extract_features_from_file(file_path)
        if res: count += 1
            
    print(f"Batch processing completed. {count}/{len(files_to_process)} files processed successfully.")

# ==========================================
# 4. 主程式入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Subspace Features (Smart Batch Mode)")
    parser.add_argument('--path', type=str, required=True, help="Input file path OR directory path")
    
    args = parser.parse_args()
    input_path = args.path
    
    if os.path.isdir(input_path):
        # 如果輸入是資料夾，進入批次模式
        print(f"--- Batch Mode Detected ---")
        process_batch(input_path)
    elif os.path.isfile(input_path):
        # 如果輸入是檔案，處理單一檔案
        print(f"--- Single File Mode Detected ---")
        saved = extract_features_from_file(input_path)
        if saved:
            print(f"Success! Saved to: {saved}")
    else:
        print(f"Error: Path does not exist: {input_path}")
