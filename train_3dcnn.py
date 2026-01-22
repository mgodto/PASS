import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import glob

from src.video_dataset import VideoDataset
from src.cnn.cnn_models import Simple3DCNN
from src.cnn.cnn_engine import train_one_epoch, evaluate
from src.config import RAW_VIDEO_DIR
from sklearn.utils.class_weight import compute_class_weight


def get_label_from_filename(filename: str) -> str:
    """從檔名中識別並回傳疾病標籤"""
    possible_labels = ["ASD", "LCS", "DHS", "HipOA"]
    for label in possible_labels:
        if label in filename:
            return label
    return None

def main(args):
    torch.manual_seed(43)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"3DCNN_lr{args.lr}_bs{args.batch_size}_{timestamp}"
    output_dir = os.path.join('results', experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"本次實驗結果將儲存於: {output_dir}")

    # 1. 準備檔案路徑和標籤
    print("掃描影片檔案...")
    all_video_paths = glob.glob(os.path.join(RAW_VIDEO_DIR, '*'))
    file_paths = []
    labels_str = []
    for path in all_video_paths:
        label = get_label_from_filename(os.path.basename(path))
        if label:
            file_paths.append(path)
            labels_str.append(label)

    le = LabelEncoder()
    labels = le.fit_transform(labels_str)
    num_classes = len(le.classes_)

    # 2. 建立數據集
    file_paths = sorted(file_paths)
    labels = np.array(labels)  # 與 file_paths 對齊
    dataset = VideoDataset(file_paths, labels, num_frames=args.num_frames)

    # ---- 3) 用固定 seed 的 random_split 切 80/20 ----
    g = torch.Generator().manual_seed(43)   # 與前面 torch.manual_seed(43) 一致
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)

    # ---- 4) 依 Subset 的 indices 回推「當次訓練用的檔名與標籤」並存檔 ----
    train_idx = train_dataset.indices
    test_idx  = test_dataset.indices

    split_json = {
        "label_encoder_classes": le.classes_.tolist(),
        "train_paths": [file_paths[i] for i in train_idx],
        "train_labels": [int(labels[i]) for i in train_idx],
        "test_paths":  [file_paths[i] for i in test_idx],
        "test_labels": [int(labels[i]) for i in test_idx],
        "seed": 43,
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "split_80_20.json"), "w", encoding="utf-8") as f:
        json.dump(split_json, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved split to {os.path.join(output_dir,'split_80_20.json')}")

    ####
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. 建立模型
    print("Building 3D-CNN model...")
    model = Simple3DCNN(num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)



    # 5. 訓練循環
    best_accuracy = 0.0
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc, report, cm_fig = evaluate(model, test_loader, loss_fn, device, le)
        
        # --- ★★★ 新增：記錄學習曲線數據 ★★★ ---
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1} Results: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with accuracy: {best_accuracy:.2%}")
        
        report_save_path = os.path.join(output_dir, f'epoch_{epoch+1}_report.txt')
        with open(report_save_path, 'w') as f:
            f.write(report)
            
        cm_save_path = os.path.join(output_dir, f'epoch_{epoch+1}_cm.png')
        cm_fig.savefig(cm_save_path)
        plt.close(cm_fig)

    # 6. 繪製並儲存學習曲線
        if train_losses and test_losses and train_accs and test_accs:
            epochs_range = range(1, args.epochs + 1)
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, train_losses, label='Training Loss')
            plt.plot(epochs_range, test_losses, label='Testing Loss')
            plt.title('Loss vs. Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, train_accs, label='Training Accuracy')
            plt.plot(epochs_range, test_accs, label='Testing Accuracy')
            plt.title('Accuracy vs. Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            learning_curve_save_path = os.path.join(output_dir, 'learning_curves.png')
            plt.savefig(learning_curve_save_path)
            print(f"Learning curves saved to {learning_curve_save_path}")
            plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D-CNN Training Script for Gait Analysis')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to sample from each video')
    args = parser.parse_args()
    main(args)
    
