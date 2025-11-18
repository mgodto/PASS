import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import os

class SubspaceDataset(Dataset):
    """用於MLP訓練的自定義數據集，只使用部分空間法特徵"""

    def __init__(self, subspace_features_file, labels_file):
        """
        Args:
            subspace_features_file (str): 部分空間法特徵檔案 (.npy)。
            labels_file (str): 標籤檔案 (.npy)。
        """
        if not os.path.exists(subspace_features_file):
             raise FileNotFoundError(f"Subspace features file not found: {subspace_features_file}")
        if not os.path.exists(labels_file):
             raise FileNotFoundError(f"Labels file not found: {labels_file}")

        self.features = np.load(subspace_features_file, allow_pickle=True)
        self.labels_str = np.load(labels_file, allow_pickle=True)

        # 獲取輸入特徵的維度
        self.input_dim = self.features.shape[1]

        # 編碼標籤
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels_str)
        self.num_classes = len(self.le.classes_)

        print(f"SubspaceDataset initialized:")
        print(f"  - Number of samples: {len(self.labels)}")
        print(f"  - Feature dimension: {self.input_dim}")
        print(f"  - Number of classes: {self.num_classes}")
        print(f"  - Class names: {self.le.classes_}")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature_vector = torch.from_numpy(self.features[index]).float()
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return feature_vector, label
