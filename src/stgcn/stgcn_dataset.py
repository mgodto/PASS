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
        # 如果檔案不存在，這裡僅印出警告，後續程式碼會使用預設值 0
        # 這對於 partition_fusion 模式是可以接受的，因為它不依賴這個檔案
        print(f"警告：找不到特徵長度資訊檔案 '{info_path}'。將使用預設值 0。")
    return lengths['max_len_first'], lengths['max_len_second']
# --- 修正結束 ---

class GaitDataset(Dataset):
    """用於ST-GCN訓練的自定義數據集，支援消融研究與多種融合模式"""

    def __init__(self, stgcn_paths_file, labels_file, subspace_features_file,
                 mode='baseline', max_len=300, fusion_features='both',
                 partition_features_dir=None):
        """
        Args:
            stgcn_paths_file (str): 骨架數據路徑檔案 (.npy)。
            labels_file (str): 標籤檔案 (.npy)。
            subspace_features_file (str): (舊版) 部分空間法特徵檔案 (.npy)。
            mode (str): 'baseline', 'late_fusion', 或 'partition_fusion'。
            max_len (int): 骨架序列的最大長度。
            fusion_features (str): 在 late_fusion 模式下要使用的特徵 ('first', 'second', 'both')。
            partition_features_dir (str): (新版) 存放 14維特徵 (.npy) 的資料夾路徑。
        """
        print(f"Initializing GaitDataset in mode: {mode}")
        if mode == 'late_fusion':
            print(f"Using fusion_features: {fusion_features}")

        if not os.path.exists(stgcn_paths_file):
             raise FileNotFoundError(f"STGCN paths file not found: {stgcn_paths_file}")
        if not os.path.exists(labels_file):
             raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        # 僅在舊版 late_fusion 模式下檢查 subspace_features_file
        if mode == 'late_fusion' and not os.path.exists(subspace_features_file):
             raise FileNotFoundError(f"Subspace features file not found: {subspace_features_file}")

        self.stgcn_paths = np.load(stgcn_paths_file, allow_pickle=True)
        self.labels_str = np.load(labels_file, allow_pickle=True)
        self.mode = mode
        self.max_len = max_len
        self.fusion_features = fusion_features
        self.partition_features_dir = partition_features_dir

        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels_str)
        self.num_classes = len(self.le.classes_)

        # --- 舊版 Fusion 變數初始化 ---
        self.all_subspace_features = None
        # --- ★★★ 核心修正點 2：讀取特徵長度資訊 ★★★ ---
        self.max_len_first, self.max_len_second = read_feature_lengths()
        self.num_total_subspace_features = self.max_len_first + self.max_len_second
        # --- 修正結束 ---

        # 載入舊版特徵 (如果存在)
        if os.path.exists(subspace_features_file):
            self.all_subspace_features = np.load(subspace_features_file, allow_pickle=True)
            # 驗證讀取的特徵維度是否與計算的總長度一致
            if self.all_subspace_features is not None and self.all_subspace_features.shape[1] != self.num_total_subspace_features:
                 # 僅警告，不中斷，避免影響不需要此特徵的模式
                 print(f"警告：讀取的 svm_features.npy 維度 ({self.all_subspace_features.shape[1]}) 與計算出的總長度 ({self.num_total_subspace_features}) 不符。")
                 if mode == 'late_fusion':
                     print("這可能會影響 late_fusion 模式的執行。")

        # 根據 mode 設定 num_selected_subspace_features
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
        
        # --- ★★★ 新增：新版 Partition Fusion 邏輯 ★★★ ---
        elif self.mode == 'partition_fusion':
            if not self.partition_features_dir:
                # 若未提供，給予預設值 (防呆)
                self.partition_features_dir = "/Users/gaoji/projects/human_gait/results/partition_npy"
            
            if not os.path.exists(self.partition_features_dir):
                print(f"嚴重警告: 找不到 Partition 特徵資料夾: {self.partition_features_dir}")
                # 這裡不 raise error，允許程式繼續執行，但在 getitem 時會補 0
            
            self.num_selected_subspace_features = 14 # 固定為 14 維
            print(f"Mode 'partition_fusion' active. Feature input dim: {self.num_selected_subspace_features}")
            
        else:
            self.num_selected_subspace_features = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 1. 讀取骨架數據
        try:
            skeleton_data = np.load(self.stgcn_paths[index])
        except FileNotFoundError:
             print(f"錯誤：在索引 {index} 處找不到骨架檔案，路徑: {self.stgcn_paths[index]}")
             # 回傳全 0 數據以避免崩潰
             return torch.zeros((3, self.max_len, 33)), torch.zeros((self.num_selected_subspace_features,)), torch.tensor(-1, dtype=torch.long)
        except Exception as e:
             print(f"載入索引 {index} 的骨架檔案時發生錯誤: {e}")
             return torch.zeros((3, self.max_len, 33)), torch.zeros((self.num_selected_subspace_features,)), torch.tensor(-1, dtype=torch.long)

        # 2. 骨架數據填充與截斷 (Padding / Truncating)
        num_frames = skeleton_data.shape[0]
        if num_frames < self.max_len:
            padding = np.zeros((self.max_len - num_frames, 33, 3))
            skeleton_data = np.concatenate([skeleton_data, padding], axis=0)
        elif num_frames > self.max_len:
            skeleton_data = skeleton_data[:self.max_len, :, :]

        # 轉為 Tensor 並調整維度 (T, V, C) -> (C, T, V) for ST-GCN
        tensor_data = torch.from_numpy(skeleton_data).float().permute(2, 0, 1)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        # 3. 根據模式回傳對應的特徵
        
        # --- Mode 1: 舊版 Late Fusion ---
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
                      # print(f"警告：選取的特徵維度 ({selected_subspace_feature_np.shape[0]}) 與預期 ({self.num_selected_subspace_features}) 不符，索引 {index}。返回零向量。")
                      selected_subspace_feature = torch.zeros((self.num_selected_subspace_features,))
                 else:
                      selected_subspace_feature = torch.from_numpy(selected_subspace_feature_np).float()
                 # --- 修正結束 ---

            return tensor_data, selected_subspace_feature, label

        # --- Mode 2: 新版 Partition Fusion (Naïve) ---
        elif self.mode == 'partition_fusion':
            # 根據骨架檔名動態尋找對應的特徵檔
            # stgcn_path 範例: .../data/processed_skeletons/20160120_ASD_lat__V1-0001.npy
            skeleton_path = self.stgcn_paths[index]
            filename = os.path.basename(skeleton_path)
            feature_filename = filename.replace('.npy', '_subspace_features.npy')
            feature_path = os.path.join(self.partition_features_dir, feature_filename)
            
            try:
                # 讀取特徵，預期形狀: (Frames, 14)
                feat_seq = np.load(feature_path)
                
                # --- Naïve Pooling ---
                # 因為目前是第一階段實驗，我們簡單地對時間軸取平均，將 (T, 14) 壓縮為 (14,)
                # 注意：這裡直接取 mean 會包含 padding 的 0，但作為 baseline 實驗是可接受的
                if feat_seq.shape[0] > 0:
                    feature_vector = np.mean(feat_seq, axis=0) 
                else:
                    feature_vector = np.zeros(14)

                selected_subspace_feature = torch.from_numpy(feature_vector).float()
                
            except Exception as e:
                # 如果讀取失敗 (例如檔案不存在)，回傳全 0 向量
                # print(f"Warning: Failed to load partition feature for {filename}: {e}")
                selected_subspace_feature = torch.zeros((14,), dtype=torch.float)

            return tensor_data, selected_subspace_feature, label

        # --- Mode 3: Baseline (無融合) ---
        else: 
            return tensor_data, label