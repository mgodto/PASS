import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os

class VideoDataset(Dataset):
    """用於3D-CNN訓練的自定義影片數據集"""

    def __init__(self, file_paths, labels, num_frames=16, frame_size=(112, 112)):
        """
        Args:
            file_paths (list): 影片檔案的路徑列表。
            labels (list): 對應的標籤列表。
            num_frames (int): 從每個影片中採樣的影格數量。
            frame_size (tuple): 每個影格要被縮放到的尺寸 (高度, 寬度)。
        """
        self.file_paths = file_paths
        self.labels = labels
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # 定義影格的轉換流程
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.frame_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        video_path = self.file_paths[index]
        label = self.labels[index]
        
        # 使用OpenCV讀取影片
        cap = cv2.VideoCapture(video_path)
        
        # 使用 try...finally 來確保 release() 一定會被呼叫
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 增加一個檢查，防止影片無法讀取或為空
            if total_frames == 0:
                print(f"Warning: Could not read frames from {video_path}. It might be corrupted or empty. Returning zeros.")
                # 返回一個全零的張量和標籤
                video_tensor = torch.zeros((3, self.num_frames, *self.frame_size))
                return video_tensor, label

            frames = []
            
            # 均勻地選擇 num_frames 個影格的索引
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # OpenCV讀取的是BGR格式，轉換為RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(self.transform(frame_rgb))
                else:
                    # 如果讀取失敗，用一個零張量代替
                    frames.append(torch.zeros((3, *self.frame_size)))
            
            # 將影格列表堆疊成一個張量
            video_tensor = torch.stack(frames, dim=1) # Shape: (C, D, H, W)
            
            return video_tensor, label

        finally:
            # 無論 try 區塊是成功完成還是中途出錯，這行都保證會被執行
            cap.release()
