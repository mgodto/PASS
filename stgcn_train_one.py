import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# 確保能從src資料夾導入我們需要的模組
from src.stgcn.stgcn_dataset import GaitDataset
from src.stgcn.stgcn_models import STGCN_Baseline, STGCN_LateFusion, STGCN_PartitionFusion
from src.stgcn.stgcn_engine import train_one_epoch, evaluate
from src.config import SVM_FEATURES_PATH, LABELS_PATH, STGCN_PATHS_PATH, PARTITION_NPY_DIR

from sklearn.utils.class_weight import compute_class_weight


def main(args):
    # 設定隨機種子以確保數據分割的一致性
    torch.manual_seed(42)

    # 檢查是否有可用的 Apple Silicon GPU (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: mps (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"Using device: cpu")

    # 根據時間戳和參數（包括 fusion_features）生成實驗資料夾名稱
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fusion_tag = f"_fusion-{args.fusion_features}" if args.model == 'late_fusion' else ""
    if args.model == 'partition_fusion':
        fusion_tag = "_partition-naive"
    # --- ★★★ 新增：在實驗名稱中加入是否使用類別權重的標記 ★★★ ---
    weight_tag = "_weighted" if args.use_class_weights else ""
    experiment_name = f"{args.model}{fusion_tag}{weight_tag}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
    output_dir = os.path.join('results', experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"本次實驗結果將儲存於: {output_dir}")
    # --- 修正結束 ---

    # 數據集準備
    print(f"Loading dataset in '{args.model}' mode")
    if args.model == 'late_fusion':
        print(f"Using fusion features: '{args.fusion_features}'...")
    try:
        dataset = GaitDataset(
            stgcn_paths_file=STGCN_PATHS_PATH,
            labels_file=LABELS_PATH,
            subspace_features_file=SVM_FEATURES_PATH,
            mode=args.model,
            max_len=args.max_len,
            fusion_features=args.fusion_features,  # <-- 將參數傳遞給 Dataset
            # ★★★ 傳入新特徵路徑
            partition_features_dir=PARTITION_NPY_DIR
        )
    except FileNotFoundError as e:
         print(f"Error initializing dataset: {e}")
         return # 數據文件缺失則退出
    except Exception as e:
         print(f"An unexpected error occurred during dataset initialization: {e}")
         return


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    

    # 模型建立
    print("Building model...")
    num_selected_subspace_features = dataset.num_selected_subspace_features
    print(f"Number of subspace features passed to the model: {num_selected_subspace_features}")

    if args.model == 'baseline':
        model = STGCN_Baseline(num_classes=dataset.num_classes).to(device)
    elif args.model == 'late_fusion':
        model = STGCN_LateFusion(
            num_subspace_features=num_selected_subspace_features,
            num_classes=dataset.num_classes
        ).to(device)
    # ★★★ 新增 Partition Fusion 分支
    elif args.model == 'partition_fusion':
        print("Initializing STGCN_PartitionFusion with 14 extra features...")
        model = STGCN_PartitionFusion(
            num_classes=dataset.num_classes,
            subspace_dim=14
        ).to(device)
    else:
        raise ValueError("Invalid model type specified.")

    # --- ★★★ 核心修正點：重新加入計算 Class Weight 的邏輯 ★★★ ---

    class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(dataset.labels),
    y=dataset.labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    if args.use_class_weights:
        print("Calculating class weights...")
        # # 直接從完整的 dataset 中獲取所有標籤來計算權重
        # labels_array = dataset.labels
        # class_counts = np.bincount(labels_array, minlength=dataset.num_classes)
        
        # # 避免除以零（如果某個類別完全不存在於數據集中）
        # class_weights_raw = 1. / (class_counts + 1e-6) # 加一個小數避免除以零
        
        # # 正規化權重 (可選，但有助於穩定訓練)
        # class_weights_normalized = class_weights_raw / np.sum(class_weights_raw) * dataset.num_classes
        
        # class_weights = torch.tensor(class_weights_normalized, dtype=torch.float).to(device)
        print(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("Not using class weights.")
        criterion = nn.CrossEntropyLoss()
    # --- 修正結束 ---

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # 訓練循環
    best_accuracy = 0.0
    # --- ★★★ 新增：用於繪製學習曲線的列表 ★★★ ---
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    # --- 新增結束 ---

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.model)
        test_loss, test_acc, _test_f1, report, cm_fig = evaluate(
            model, test_loader, criterion, device, dataset.le, args.model
        )
        
        print(f"Epoch {epoch+1} Results: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")
        
        # --- ★★★ 新增：記錄學習曲線數據 ★★★ ---
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        # --- 新增結束 ---

        # 儲存最佳模型等結果到實驗資料夾
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

    # --- ★★★ 新增：繪製並儲存學習曲線 ★★★ ---
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
    # --- 新增結束 ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ST-GCN Training Script for Gait Analysis with Ablation Study')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['baseline', 'late_fusion', 'partition_fusion'], 
                        help='Model type')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=300, help='Maximum sequence length for padding/truncating')
    parser.add_argument('--fusion_features', type=str, default='both', choices=['first', 'second', 'both'], 
                        help="Which subspace features to use in late_fusion mode ('first', 'second', or 'both')")
    # --- ★★★ 新增：控制是否使用類別權重的參數 ★★★ ---
    parser.add_argument('--use_class_weights', default=True, action='store_true', 
                        help='Apply class weighting to the loss function to handle imbalance.')
    # --- 新增結束 ---
    args = parser.parse_args()
    main(args)
