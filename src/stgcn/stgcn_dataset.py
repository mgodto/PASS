# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from sklearn.preprocessing import LabelEncoder

# class GaitDataset(Dataset):
#     """用於ST-GCN訓練的自定義數據集"""
    
#     # --- ★★★ 這就是修正的部分 ★★★ ---
#     # 將 __init__ 函數的參數名從 stgcn_paths 改為 stgcn_paths_file
#     def __init__(self, stgcn_paths_file, labels_file, subspace_features_file, mode='baseline', max_len=300):
#         # --- 修正結束 ---
        
#         self.stgcn_paths = np.load(stgcn_paths_file, allow_pickle=True)
#         self.labels_str = np.load(labels_file, allow_pickle=True)
#         self.mode = mode
#         self.max_len = max_len

#         # 編碼標籤
#         self.le = LabelEncoder()
#         self.labels = self.le.fit_transform(self.labels_str)
#         self.num_classes = len(self.le.classes_)
        
#         # 如果是後期融合模式，則載入部分空間法特徵
#         self.subspace_features = None
#         if self.mode == 'late_fusion':
#             self.subspace_features = np.load(subspace_features_file, allow_pickle=True)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         # 載入骨架 .npy 檔案
#         skeleton_data = np.load(self.stgcn_paths[index])
        

#         # ★★★ 新增：翻轉 Z 軸座標 ★★★
#         # 假設數據形狀為 (Frames, Joints, Coords), Coords 是 (x, y, z)
#         # 我們要將第 3 個座標 (索引為 2) 的值乘以 -1
#         skeleton_data[:, :, 2] = -skeleton_data[:, :, 2]

#         # 填充與截斷
#         num_frames = skeleton_data.shape[0]
#         if num_frames < self.max_len:
#             padding = np.zeros((self.max_len - num_frames, 33, 3))
#             skeleton_data = np.concatenate([skeleton_data, padding], axis=0)
#         elif num_frames > self.max_len:
#             skeleton_data = skeleton_data[:self.max_len, :, :]
        
#         ######################################### 正規化 (每個維度獨立正規化) ################################
#         # mean = skeleton_data.mean(axis=(0, 1), keepdims=True)
#         # std = skeleton_data.std(axis=(0, 1), keepdims=True)
#         # std[std == 0] = 1e-6 # 避免除以零
#         # skeleton_data = (skeleton_data - mean) / std
#         ####################################################################################################

#         # 轉換為 PyTorch Tensor (C, T, V)
#         tensor_data = torch.from_numpy(skeleton_data).float().permute(2, 0, 1)
        
#         label = torch.tensor(self.labels[index], dtype=torch.long)
        
#         if self.mode == 'late_fusion':
#             subspace_feature = torch.from_numpy(self.subspace_features[index]).float()
#             return tensor_data, subspace_feature, label
#         else: # baseline 模式
#             return tensor_data, label


import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import os

# --- ★★★ 核心修正點 1：定義讀取長度資訊的函數 ★★★ ---
def read_feature_lengths(info_path='results/train/feature_lengths.txt'):
    """從檔案中讀取 max_len_first 和 max_len_second"""
    lengths = {'max_len_first': 0, 'max_len_second': 0}
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=')
                    if key in lengths:
                        try:
                            lengths[key] = int(value)
                        except ValueError:
                            print(f"警告：無法解析 '{info_path}' 中的 '{key}' 值：'{value}'")
    else:
        print(f"警告：找不到特徵長度資訊檔案 '{info_path}'。將使用預設值 0。")
    return lengths['max_len_first'], lengths['max_len_second']
# --- 修正結束 ---

class GaitDataset(Dataset):
    """用於ST-GCN訓練的自定義數據集，支援消融研究"""

    def __init__(self, stgcn_paths_file, labels_file, subspace_features_file,
                 mode='baseline', max_len=300, fusion_features='both'):
        """
        Args:
            stgcn_paths_file (str): 骨架數據路徑檔案 (.npy)。
            labels_file (str): 標籤檔案 (.npy)。
            subspace_features_file (str): 部分空間法特徵檔案 (.npy)。
            mode (str): 'baseline' 或 'late_fusion'。
            max_len (int): 骨架序列的最大長度。
            fusion_features (str): 在 late_fusion 模式下要使用的特徵 ('first', 'second', 'both')。
        """
        print(f"Initializing GaitDataset in mode: {mode}")
        if mode == 'late_fusion':
            print(f"Using fusion_features: {fusion_features}")

        if not os.path.exists(stgcn_paths_file):
             raise FileNotFoundError(f"STGCN paths file not found: {stgcn_paths_file}")
        if not os.path.exists(labels_file):
             raise FileNotFoundError(f"Labels file not found: {labels_file}")
        if mode == 'late_fusion' and not os.path.exists(subspace_features_file):
             raise FileNotFoundError(f"Subspace features file not found: {subspace_features_file}")

        self.stgcn_paths = np.load(stgcn_paths_file, allow_pickle=True)
        self.labels_str = np.load(labels_file, allow_pickle=True)
        self.mode = mode
        self.max_len = max_len
        self.fusion_features = fusion_features

        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels_str)
        self.num_classes = len(self.le.classes_)

        self.all_subspace_features = None
        # --- ★★★ 核心修正點 2：讀取特徵長度資訊 ★★★ ---
        self.max_len_first, self.max_len_second = read_feature_lengths()
        self.num_total_subspace_features = self.max_len_first + self.max_len_second
        # --- 修正結束 ---

        if os.path.exists(subspace_features_file):
            self.all_subspace_features = np.load(subspace_features_file, allow_pickle=True)
            # 驗證讀取的特徵維度是否與計算的總長度一致
            if self.all_subspace_features is not None and self.all_subspace_features.shape[1] != self.num_total_subspace_features:
                 print(f"警告：讀取的 svm_features.npy 維度 ({self.all_subspace_features.shape[1]}) 與計算出的總長度 ({self.num_total_subspace_features}) 不符。請重新執行 build_master_dataset.py。")
                 # 可以選擇拋出錯誤或繼續（但可能會有問題）
                 # raise ValueError("Feature dimension mismatch.")
                 self.all_subspace_features = None # 設為 None 以觸發後續錯誤處理

        # 根據 fusion_features 決定實際使用的特徵數量
        if self.mode == 'late_fusion':
            if self.fusion_features == 'first':
                self.num_selected_subspace_features = self.max_len_first
            elif self.fusion_features == 'second':
                self.num_selected_subspace_features = self.max_len_second
            elif self.fusion_features == 'both':
                self.num_selected_subspace_features = self.num_total_subspace_features
            else:
                print(f"警告：無效的 fusion_features 值 '{self.fusion_features}'。將預設為 'both'。")
                self.fusion_features = 'both'
                self.num_selected_subspace_features = self.num_total_subspace_features
        else:
            self.num_selected_subspace_features = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        try:
            skeleton_data = np.load(self.stgcn_paths[index])
        except FileNotFoundError:
             print(f"錯誤：在索引 {index} 處找不到骨架檔案，路徑: {self.stgcn_paths[index]}")
             return torch.zeros((3, self.max_len, 33)), torch.zeros((self.num_selected_subspace_features,)), torch.tensor(-1, dtype=torch.long)
        except Exception as e:
             print(f"載入索引 {index} 的骨架檔案時發生錯誤: {e}")
             return torch.zeros((3, self.max_len, 33)), torch.zeros((self.num_selected_subspace_features,)), torch.tensor(-1, dtype=torch.long)

        # 填充與截斷
        num_frames = skeleton_data.shape[0]
        if num_frames < self.max_len:
            padding = np.zeros((self.max_len - num_frames, 33, 3))
            skeleton_data = np.concatenate([skeleton_data, padding], axis=0)
        elif num_frames > self.max_len:
            skeleton_data = skeleton_data[:self.max_len, :, :]

        tensor_data = torch.from_numpy(skeleton_data).float().permute(2, 0, 1)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        if self.mode == 'late_fusion':
            if self.all_subspace_features is None:
                 print(f"錯誤：請求了部分空間法特徵，但在索引 {index} 處未載入。")
                 selected_subspace_feature = torch.zeros((self.num_selected_subspace_features,))
            else:
                 full_subspace_feature = self.all_subspace_features[index]
                 # --- ★★★ 核心修正點 3：根據 max_len_first 精確選取 ★★★ ---
                 if self.fusion_features == 'first':
                      selected_subspace_feature_np = full_subspace_feature[:self.max_len_first]
                 elif self.fusion_features == 'second':
                      selected_subspace_feature_np = full_subspace_feature[self.max_len_first:]
                 else: # 'both'
                      selected_subspace_feature_np = full_subspace_feature

                 # 再次驗證選取的維度是否正確
                 if selected_subspace_feature_np.shape[0] != self.num_selected_subspace_features:
                      print(f"警告：選取的特徵維度 ({selected_subspace_feature_np.shape[0]}) 與預期 ({self.num_selected_subspace_features}) 不符，索引 {index}。返回零向量。")
                      selected_subspace_feature = torch.zeros((self.num_selected_subspace_features,))
                 else:
                      selected_subspace_feature = torch.from_numpy(selected_subspace_feature_np).float()
                 # --- 修正結束 ---

            return tensor_data, selected_subspace_feature, label
        else: # baseline 模式
            return tensor_data, label






