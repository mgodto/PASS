# build_master_dataset.py

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# 確保能從 src 導入設定
try:
    from src.config import PROCESSED_SKELETON_DIR, SUBSPACE_NPY_DIR
except ImportError:
    print("警告：無法從 src.config 導入設定，將使用預設路徑。")
    PROCESSED_SKELETON_DIR = 'data/processed_skeletons'
    SUBSPACE_NPY_DIR = 'results/subspace_npy'

# 定義輸出的資料夾
OUTPUT_DIR = 'results/train'

def get_label_from_filename(filename: str) -> str:
    """從檔名中識別並回傳疾病標籤"""
    possible_labels = ["ASD", "LCS", "DHS", "HipOA", "Normal_Control"] # 加入 Normal_Control
    for label in possible_labels:
        if label in filename:
            return label
    print(f"警告：在檔名 '{filename}' 中找不到已知標籤。")
    return None # 如果找不到標籤，返回 None

def pad_truncate_sequence(seq: np.ndarray, max_len: int) -> np.ndarray:
    """將序列填充或截斷到指定的max_len"""
    current_len = seq.shape[0]
    if current_len > max_len:
        return seq[:max_len]
    elif current_len < max_len:
        padding_shape = (max_len - current_len,) + seq.shape[1:]
        padding = np.zeros(padding_shape, dtype=seq.dtype)
        return np.concatenate([seq, padding], axis=0)
    else:
        return seq

def main():
    print("--- 開始建立最終訓練資料集 (包含填充/截斷) ---")

    skeleton_files = glob.glob(os.path.join(PROCESSED_SKELETON_DIR, '*.npy'))
    if not skeleton_files:
        print(f"錯誤：在 {PROCESSED_SKELETON_DIR} 中找不到已處理的骨架檔案。")
        return

    skeleton_files.sort()
    print(f"找到 {len(skeleton_files)} 個唯一的骨架樣本。")

    # --- 第一步：遍歷所有檔案，找到 first 和 second 特徵各自的最大長度 ---
    max_len_first = 0
    max_len_second = 0
    valid_files_info = [] # 儲存有效的檔案路徑和標籤

    print("計算 first 和 second 特徵的最大長度...")
    for skeleton_path in tqdm(skeleton_files):
        video_id = os.path.splitext(os.path.basename(skeleton_path))[0]
        label = get_label_from_filename(video_id)

        if label is None:
            continue

        path_first = os.path.join(SUBSPACE_NPY_DIR, f"{video_id}_first_mag.npy")
        path_second = os.path.join(SUBSPACE_NPY_DIR, f"{video_id}_second_mag.npy")

        if os.path.exists(path_first) and os.path.exists(path_second):
            try:
                f1 = np.atleast_1d(np.load(path_first))
                # --- ★★★ 核心修正點 1：讀取 second 特徵 ★★★ ---
                f2 = np.atleast_1d(np.load(path_second))

                # 更新各自的最大長度
                if len(f1) > max_len_first:
                    max_len_first = len(f1)
                # --- ★★★ 核心修正點 2：更新 second 的最大長度 ★★★ ---
                if len(f2) > max_len_second:
                    max_len_second = len(f2)

                valid_files_info.append({
                    'skeleton_path': skeleton_path,
                    'label': label,
                    'video_id': video_id,
                    'path_first': path_first,
                    'path_second': path_second
                })
            except Exception as e:
                print(f"讀取檔案時發生錯誤 {video_id}: {e}")
        else:
             print(f"警告：缺少部分空間法特徵檔案 {video_id}")


    if max_len_first == 0 or max_len_second == 0:
        print("錯誤：無法確定最大長度。找不到有效的 first 或 second 特徵檔案。")
        return
    print(f"First 特徵的最大長度 (max_len_first): {max_len_first}")
    print(f"Second 特徵的最大長度 (max_len_second): {max_len_second}")
    total_max_len = max_len_first + max_len_second
    print(f"拼接後的總特徵長度: {total_max_len}")

    # --- ★★★ 核心修正點 3：將最大長度資訊儲存起來，供 dataset 使用 ★★★ ---
    # 我們可以儲存一個小的 .txt 或 .npy 檔案來記錄這些值
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    len_info_path = os.path.join(OUTPUT_DIR, 'feature_lengths.txt')
    with open(len_info_path, 'w') as f:
        f.write(f"max_len_first={max_len_first}\n")
        f.write(f"max_len_second={max_len_second}\n")
    print(f"特徵長度資訊已儲存至: {len_info_path}")
    # --- 修正結束 ---


    # --- 第二步：遍歷有效的檔案，進行填充/截斷並拼接 ---
    svm_features_list = []
    labels_list = []
    stgcn_paths_list = []

    print("建立包含填充/截斷特徵的資料集...")
    for file_info in tqdm(valid_files_info):
        try:
            f1 = np.atleast_1d(np.load(file_info['path_first']))
            # --- ★★★ 核心修正點 4：讀取 second 特徵 ★★★ ---
            f2 = np.atleast_1d(np.load(file_info['path_second']))

            # 分別進行填充/截斷
            f1_padded = pad_truncate_sequence(f1, max_len_first)
            f2_padded = pad_truncate_sequence(f2, max_len_second)

            # --- ★★★ 核心修正點 5：正確拼接 ★★★ ---
            # 拼接成最終的特徵向量
            combined_feature = np.concatenate([f1_padded, f2_padded])

            if len(combined_feature) != total_max_len:
                print(f"錯誤：拼接後的長度不符 {file_info['video_id']} (預期 {total_max_len}, 實際 {len(combined_feature)})")
                continue

            svm_features_list.append(combined_feature)
            labels_list.append(file_info['label'])
            stgcn_paths_list.append(file_info['skeleton_path'])

        except Exception as e:
            print(f"處理檔案時發生錯誤 {file_info['video_id']}: {e}")

    # --- 第三步：儲存最終的訓練檔案 ---
    if svm_features_list:
        X_svm = np.array(svm_features_list)
        y_labels = np.array(labels_list)
        stgcn_paths = np.array(stgcn_paths_list)

        svm_features_save_path = os.path.join(OUTPUT_DIR, 'svm_features.npy')
        labels_save_path = os.path.join(OUTPUT_DIR, 'labels.npy')
        stgcn_paths_save_path = os.path.join(OUTPUT_DIR, 'stgcn_paths.npy')

        np.save(svm_features_save_path, X_svm)
        np.save(labels_save_path, y_labels)
        np.save(stgcn_paths_save_path, stgcn_paths)

        print("\n--- 最終資料集建立成功！ ---")
        print(f"SVM 特徵矩陣已儲存至: {svm_features_save_path} (形狀: {X_svm.shape})")
        print(f"標籤向量已儲存至: {labels_save_path} (形狀: {y_labels.shape})")
        print(f"STGCN 路徑列表已儲存至: {stgcn_paths_save_path} (數量: {len(stgcn_paths)})")
        
        unique, counts = np.unique(y_labels, return_counts=True)
        print("\n資料集標籤分佈:")
        for label, count in zip(unique, counts):
            print(f"- {label}: {count}")

    else:
        print("沒有成功聚合任何數據。請檢查輸入檔案和路徑。")

if __name__ == '__main__':
    main()

