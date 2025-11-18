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
