import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    """一個輕量級的3D-CNN模型，用於影片分類"""

    def __init__(self, num_classes=4):
        super(Simple3DCNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            # 新增一個 Conv3d 來降維，取代 Pooling
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            
            # Block 2
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            # 新增一個 Conv3d 來降維，取代 Pooling
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            # Block 3
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            # 新增一個 Conv3d 來降維，取代 Pooling
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
        )
        
        # 使用AdaptiveAvgPool3d來處理不同大小的特徵圖
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (N, C, D, H, W)
        x = self.feature_extractor(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
