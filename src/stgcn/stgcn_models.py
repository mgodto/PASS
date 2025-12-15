import torch
import torch.nn as nn
from src.stgcn.stgcn_layer import STGCNBlock

# MediaPipe 33個關節點的標準連接關係
EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], 
    [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], 
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [11, 23], [12, 24], 
    [23, 24], [23, 25], [25, 27], [27, 29], [27, 31], [24, 26], [26, 28], 
    [28, 30], [28, 32]
]

class STGCN_Baseline(nn.Module):
    """純ST-GCN模型"""
    def __init__(self, num_nodes=33, num_classes=4):
        super().__init__()
        
        edge_index_tensor = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
        self.register_buffer('edge_index', edge_index_tensor)
        
        self.st_blocks = nn.ModuleList([
            STGCNBlock(in_channels=3, out_channels=64, kernel_size=9),
            STGCNBlock(in_channels=64, out_channels=128, kernel_size=9),
            STGCNBlock(in_channels=128, out_channels=256, kernel_size=9)
        ])
        
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x): # <-- 只接收 x 作為參數
        for block in self.st_blocks:
            x = block(x, self.edge_index) # <-- 內部使用 self.edge_index
        
        x = x.mean(dim=[2, 3])
        return self.fc(x)

class STGCN_LateFusion(nn.Module):
    """帶有後期融合的ST-GCN模型"""
    def __init__(self, num_nodes=33, num_subspace_features=0, num_classes=4):
        super().__init__()
        
        edge_index_tensor = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
        self.register_buffer('edge_index', edge_index_tensor)
        
        self.st_feature_extractor = nn.ModuleList([
            STGCNBlock(in_channels=3, out_channels=64, kernel_size=9),
            STGCNBlock(in_channels=64, out_channels=128, kernel_size=9),
            STGCNBlock(in_channels=128, out_channels=256, kernel_size=9)
        ])
        
        st_gcn_output_dim = 256
        
        fused_dim = st_gcn_output_dim + num_subspace_features
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, skeleton_data, subspace_features): # <-- 只接收骨架和特徵
        x = skeleton_data
        for block in self.st_feature_extractor:
            x = block(x, self.edge_index) # <-- 內部使用 self.edge_index

        skeleton_features = x.mean(dim=[2, 3])
        fused_vector = torch.cat([skeleton_features, subspace_features], dim=1)
        return self.fusion_head(fused_vector)


class STGCN_PartitionFusion(nn.Module):
    """
    針對新版 14維 Partition 特徵的 Naïve Fusion 模型
    直接將 256維 ST-GCN 特徵與 14維 物理特徵串接
    """
    def __init__(self, num_nodes=33, num_classes=4, subspace_dim=14):
        super().__init__()
        
        edge_index_tensor = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
        self.register_buffer('edge_index', edge_index_tensor)
        
        # ST-GCN 主幹
        self.st_feature_extractor = nn.ModuleList([
            STGCNBlock(in_channels=3, out_channels=64, kernel_size=9),
            STGCNBlock(in_channels=64, out_channels=128, kernel_size=9),
            STGCNBlock(in_channels=128, out_channels=256, kernel_size=9)
        ])
        
        st_gcn_output_dim = 256
        
        # 融合層：256 + 14 = 270
        fused_dim = st_gcn_output_dim + subspace_dim
        
        # Naïve Fusion Head (直接過 FC)
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(), # 可以加個非線性
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, skeleton_data, subspace_features):
        # 1. 骨架特徵提取
        x = skeleton_data
        for block in self.st_feature_extractor:
            x = block(x, self.edge_index)

        # Global Average Pooling (N, 256, T, V) -> (N, 256)
        skeleton_features = x.mean(dim=[2, 3])
        
        # 2. 特徵融合
        # subspace_features shape: (N, 14)
        fused_vector = torch.cat([skeleton_features, subspace_features], dim=1)
        
        # 3. 分類
        return self.fusion_head(fused_vector)