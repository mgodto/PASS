import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import os

# --- 輔助函數：讀取特徵長度 (舊版功能保留) ---
def read_feature_lengths(info_path='results/train/feature_lengths.txt'):
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
                            pass
    return lengths['max_len_first'], lengths['max_len_second']

class GaitDataset(Dataset):
    """用於ST-GCN訓練的自定義數據集"""

    def _infer_subspace_feature_dim(self):
        if self.all_subspace_features is None:
            return None
        feats = np.asarray(self.all_subspace_features)
        if feats.ndim >= 2:
            return int(feats.shape[1])
        if feats.ndim == 1 and len(feats) > 0:
            first = np.asarray(feats[0])
            if first.ndim >= 1:
                return int(first.shape[0])
        return None

    def __init__(self, stgcn_paths_file, labels_file, subspace_features_file,
                 mode='baseline', max_len=300, fusion_features='both',
                 partition_features_dir=None): # ★★★ 新增參數
        """
        Args:
            partition_features_dir (str): 存放新版 14維特徵 (.npy) 的資料夾路徑。
        """
        print(f"Initializing GaitDataset in mode: {mode}")
        
        # 基本路徑檢查
        if not os.path.exists(stgcn_paths_file):
             raise FileNotFoundError(f"STGCN paths file not found: {stgcn_paths_file}")
        if not os.path.exists(labels_file):
             raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        # 載入基礎數據
        self.stgcn_paths = np.load(stgcn_paths_file, allow_pickle=True)
        self.labels_str = np.load(labels_file, allow_pickle=True)
        self.mode = mode
        self.max_len = max_len
        self.fusion_features = fusion_features
        self.partition_features_dir = partition_features_dir

        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels_str)
        self.num_classes = len(self.le.classes_)

        # --- 舊版 Late Fusion 初始化 (保持不變) ---
        self.all_subspace_features = None
        self.max_len_first, self.max_len_second = read_feature_lengths()
        self.num_total_subspace_features = self.max_len_first + self.max_len_second
        self.num_selected_subspace_features = 0 # 預設

        if self.mode == 'late_fusion':
            if os.path.exists(subspace_features_file):
                self.all_subspace_features = np.load(subspace_features_file, allow_pickle=True)
            
            if self.fusion_features == 'first':
                self.num_selected_subspace_features = self.max_len_first
            elif self.fusion_features == 'second':
                self.num_selected_subspace_features = self.max_len_second
            else:
                self.num_selected_subspace_features = self.num_total_subspace_features

            feature_dim = self._infer_subspace_feature_dim()
            if feature_dim is not None:
                expected_dim = self.max_len_first + self.max_len_second
                if expected_dim == 0 and self.fusion_features in ('both', 'second'):
                    self.num_total_subspace_features = feature_dim
                    self.num_selected_subspace_features = feature_dim
                elif expected_dim > 0 and expected_dim != feature_dim:
                    if self.max_len_first > feature_dim:
                        self.max_len_first = feature_dim
                    self.max_len_second = max(feature_dim - self.max_len_first, 0)
                    self.num_total_subspace_features = self.max_len_first + self.max_len_second
                    if self.fusion_features == 'first':
                        self.num_selected_subspace_features = self.max_len_first
                    elif self.fusion_features == 'second':
                        self.num_selected_subspace_features = self.max_len_second
                    else:
                        self.num_selected_subspace_features = self.num_total_subspace_features

        # --- ★★★ 新版 Partition Fusion 初始化 ★★★ ---
        elif self.mode == 'partition_fusion':
            if not self.partition_features_dir:
                # 預設路徑防呆
                self.partition_features_dir = "/Users/gaoji/projects/human_gait/results/partition_npy"
            
            if not os.path.exists(self.partition_features_dir):
                print(f"嚴重警告: 找不到 Partition 特徵資料夾: {self.partition_features_dir}")
            
            # ★★★ 修改：特徵維度變為 42 (14部位 * 3統計量) ★★★
            self.num_selected_subspace_features = 14 * 3 
            print(f"Mode 'partition_fusion' active. Input Feature Dim: {self.num_selected_subspace_features} (14 parts x 3 stats)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 1. 讀取骨架數據 (保持不變)
        try:
            skeleton_data = np.load(self.stgcn_paths[index])
        except Exception:
            # 回傳全 0 以避免崩潰 (根據 mode 回傳不同維度的特徵占位符)
            if self.mode in ('late_fusion', 'partition_fusion'):
                feat_dim = self.num_selected_subspace_features if self.num_selected_subspace_features > 0 else 1
                return torch.zeros((3, self.max_len, 33)), torch.zeros((feat_dim,)), torch.tensor(-1, dtype=torch.long)
            return torch.zeros((3, self.max_len, 33)), torch.tensor(-1, dtype=torch.long)

        # Padding / Truncating (保持不變)
        num_frames = skeleton_data.shape[0]
        if num_frames < self.max_len:
            padding = np.zeros((self.max_len - num_frames, 33, 3))
            skeleton_data = np.concatenate([skeleton_data, padding], axis=0)
        elif num_frames > self.max_len:
            skeleton_data = skeleton_data[:self.max_len, :, :]

        tensor_data = torch.from_numpy(skeleton_data).float().permute(2, 0, 1)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        # 2. 處理特徵回傳
        
        # --- Mode: 舊版 Late Fusion ---
        if self.mode == 'late_fusion':
            if self.all_subspace_features is None:
                 selected_subspace_feature = torch.zeros((self.num_selected_subspace_features,))
            else:
                 full_subspace_feature = self.all_subspace_features[index]
                 if self.fusion_features == 'first':
                      selected_subspace_feature_np = full_subspace_feature[:self.max_len_first]
                 elif self.fusion_features == 'second':
                      selected_subspace_feature_np = full_subspace_feature[self.max_len_first:]
                 else:
                      selected_subspace_feature_np = full_subspace_feature
                 selected_subspace_feature = torch.from_numpy(selected_subspace_feature_np).float()
            return tensor_data, selected_subspace_feature, label

        # --- Mode: 新版 Partition Fusion (Statistical Pooling) ---
        elif self.mode == 'partition_fusion':
            filename = os.path.basename(self.stgcn_paths[index])
            feature_filename = filename.replace('.npy', '_subspace_features.npy')
            feature_path = os.path.join(self.partition_features_dir, feature_filename)
            
            try:
                # 讀取特徵 (Frames, 14)
                feat_seq = np.load(feature_path)
                
                if feat_seq.shape[0] > 0:
                    # ★★★ 核心修改：統計池化 (Statistical Pooling) ★★★
                    # 1. Max: 捕捉異常峰值 (Peak)
                    feat_max = np.max(feat_seq, axis=0)
                    # 2. Mean: 捕捉整體水平
                    feat_mean = np.mean(feat_seq, axis=0)
                    # 3. Std: 捕捉晃動程度 (Variability)
                    feat_std = np.std(feat_seq, axis=0)
                    
                    # 串接: 14 + 14 + 14 = 42 維
                    feature_vector = np.concatenate([feat_max, feat_mean, feat_std])
                else:
                    feature_vector = np.zeros(self.num_selected_subspace_features) # 42維全0

                selected_subspace_feature = torch.from_numpy(feature_vector).float()
                
            except Exception as e:
                # print(f"Warning: Failed to load {feature_filename}: {e}")
                selected_subspace_feature = torch.zeros((self.num_selected_subspace_features,), dtype=torch.float)

            return tensor_data, selected_subspace_feature, label

        # --- Mode: Baseline ---
        else:
            return tensor_data, label
