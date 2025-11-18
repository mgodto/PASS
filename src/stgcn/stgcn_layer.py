import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class STGCNBlock(nn.Module):
    """
    自定義的時空間圖卷積塊 (Spatio-Temporal Graph Convolutional Block)
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(STGCNBlock, self).__init__()
        # 時間卷積層
        self.tcn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=((kernel_size - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels), # 在卷積和 ReLU 之間加入 BatchNorm 
            nn.ReLU(),
        )
        # 空間圖卷積層
        self.gcn = GCNConv(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        # x 的輸入形狀: (N, C, T, V) N=batch, C=channels, T=frames, V=nodes
        x = self.tcn(x)
        
        # 為了 GCN，需要調整形狀
        N, C, T, V = x.shape
        x_gcn = x.permute(0, 2, 3, 1).reshape(N * T, V, C) # (N*T, V, C)
        
        # 執行圖卷積
        x_gcn = self.gcn(x_gcn, edge_index)
        x_gcn = self.relu(x_gcn)
        
        # 將形狀調整回來
        x_gcn = x_gcn.reshape(N, T, V, C).permute(0, 3, 1, 2) # (N, C, T, V)
        return x_gcn.contiguous() # 確保內存連續性

