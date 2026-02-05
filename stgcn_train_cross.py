import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# 確保能從src資料夾導入我們需要的模組
from src.stgcn.stgcn_dataset import GaitDataset
from src.stgcn.stgcn_models import (
    STGCN_Baseline,
    STGCN_LateFusion,
    STGCN_PartitionFusion,
    STGCN_PartitionFusionConv,
    STGCN_PartitionFusionAttention,
)
# ★★★ 注意：這裡導入的 evaluate 已經是我們剛修改過會回傳 F1 的版本 ★★★
from src.stgcn.stgcn_engine import train_one_epoch, evaluate
from src.config import SVM_FEATURES_PATH, LABELS_PATH, STGCN_PATHS_PATH, PARTITION_NPY_DIR

from sklearn.utils.class_weight import compute_class_weight


def train_fold(fold_idx, train_dataset, test_dataset, dataset_full, args, output_dir, device):
    """
    執行單一 Fold 的訓練流程
    """
    print(f"\n{'='*20} Start Training Fold {fold_idx+1}/{args.k_folds} {'='*20}")
    
    fold_dir = os.path.join(output_dir, f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # 儲存 Indices
    np.save(os.path.join(fold_dir, 'train_indices.npy'), train_dataset.indices)
    np.save(os.path.join(fold_dir, 'test_indices.npy'), test_dataset.indices)

    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 計算 Class Weights
    train_indices = train_dataset.indices
    train_labels = dataset_full.labels[train_indices]
    
    if args.use_class_weights:
        train_classes = np.unique(train_labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=train_classes,
            y=train_labels
        )
        class_weights_full = np.ones(dataset_full.num_classes, dtype=np.float32)
        for class_id, weight in zip(train_classes, class_weights):
            class_weights_full[int(class_id)] = weight
        class_weights = torch.tensor(class_weights_full, dtype=torch.float).to(device)
        print(f"Fold {fold_idx} Class Weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # 建立模型
    num_selected_subspace_features = dataset_full.num_selected_subspace_features
    num_classes = dataset_full.num_classes

    if args.model == 'baseline':
        model = STGCN_Baseline(num_classes=num_classes).to(device)
    elif args.model == 'late_fusion':
        model = STGCN_LateFusion(
            num_subspace_features=num_selected_subspace_features,
            num_classes=num_classes
        ).to(device)
    elif args.model == 'partition_fusion':
        model = STGCN_PartitionFusion(
            num_classes=num_classes,
            subspace_dim=num_selected_subspace_features
        ).to(device)
    elif args.model == 'partition_fusion_conv':
        model = STGCN_PartitionFusionConv(
            num_classes=num_classes,
            subspace_dim=num_selected_subspace_features,
            part_feat_dim=getattr(dataset_full, "part_feature_dim", 2),
        ).to(device)
    elif args.model == 'partition_fusion_attn':
        model = STGCN_PartitionFusionAttention(
            num_classes=num_classes,
            subspace_dim=num_selected_subspace_features,
        ).to(device)
    else:
        raise ValueError("Invalid model type")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75)

    # 訓練迴圈
    best_accuracy = 0.0
    best_f1_at_best_acc = 0.0 # 用來記錄最佳 Accuracy 那一刻的 F1
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    test_f1s = [] # 記錄每個 epoch 的 F1

    for epoch in range(args.epochs):
        # ★★★ 修正解包：現在 evaluate 會回傳 f1 ★★★
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.model
        )
        test_loss, test_acc, test_f1, report, cm_fig = evaluate(model, test_loader, criterion, device, dataset_full.le, args.model)
        
        if scheduler: scheduler.step()
        
        # Log 顯示 F1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Fold {fold_idx}|Ep {epoch+1}] Loss: {train_loss:.4f} | Acc: {test_acc:.2%} | F1: {test_f1:.2%}")

        train_losses.append(train_loss); test_losses.append(test_loss)
        train_accs.append(train_acc); test_accs.append(test_acc)
        test_f1s.append(test_f1)

        # 儲存最佳模型 (依據 Accuracy)
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_f1_at_best_acc = test_f1 # 鎖定這個 Accuracy 對應的 F1
            
            save_path = os.path.join(fold_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            
            # 儲存當下的報告
            with open(os.path.join(fold_dir, 'best_report.txt'), 'w') as f:
                f.write(f"Best Epoch: {epoch+1}\n")
                f.write(f"Accuracy: {test_acc:.2%}\n")
                f.write(f"Macro F1: {test_f1:.2%}\n\n")
                f.write(report)
            cm_fig.savefig(os.path.join(fold_dir, 'best_cm.png'))
        
        plt.close(cm_fig)

    # 繪製曲線 (新增 F1 曲線)
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train'); plt.plot(test_losses, label='Test')
    plt.title('Loss'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train'); plt.plot(test_accs, label='Test')
    plt.title('Accuracy'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(test_f1s, label='Test F1', color='orange')
    plt.title('F1 Score'); plt.legend(); plt.grid(True)
    
    plt.savefig(os.path.join(fold_dir, 'learning_curve.png'))
    plt.close()

    print(f"Fold {fold_idx} Best Result -> Acc: {best_accuracy:.2%}, F1: {best_f1_at_best_acc:.2%}")
    # 回傳兩個指標
    return best_accuracy, best_f1_at_best_acc

def main(args):
    # 設定全域種子
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: mps")
    else:
        device = torch.device("cpu")
        print(f"Using device: cpu")

    # 實驗命名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fusion_tag = f"_{args.fusion_features}" if args.model == 'late_fusion' else ""
    if args.model == 'partition_fusion':
        fusion_tag = "_partition"
    elif args.model == 'partition_fusion_conv':
        fusion_tag = "_partition-conv"
    elif args.model == 'partition_fusion_attn':
        fusion_tag = "_partition-attn"

    hand_tag = ""
    if args.model in ('partition_fusion', 'partition_fusion_conv', 'partition_fusion_attn'):
        if args.partition_hand_mode != "both":
            hand_tag = f"_hands-{args.partition_hand_mode}"

    experiment_name = f"{args.model}{fusion_tag}{hand_tag}_{args.k_folds}Fold_bs{args.batch_size}_{timestamp}"
    output_dir = os.path.join('results', 'kfold_experiments', experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Experiment Results will be saved to: {output_dir}")

    # 載入資料
    print(f"Loading full dataset...")
    try:
        dataset = GaitDataset(
            stgcn_paths_file=STGCN_PATHS_PATH,
            labels_file=LABELS_PATH,
            subspace_features_file=SVM_FEATURES_PATH,
            mode=args.model,
            max_len=args.max_len,
            fusion_features=args.fusion_features,
            partition_features_dir=PARTITION_NPY_DIR,
            partition_hand_mode=args.partition_hand_mode,
        )
    except Exception as e:
         print(f"Dataset Error: {e}"); return

    # K-Fold
    print(f"Initializing {args.k_folds}-Fold Cross Validation...")
    kfold = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    
    fold_accs = []
    fold_f1s = [] # 儲存每個 fold 的 F1

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, dataset.labels)):
        train_sub = Subset(dataset, train_ids)
        test_sub = Subset(dataset, test_ids)
        
        # 接收兩個回傳值
        best_acc, best_f1 = train_fold(fold, train_sub, test_sub, dataset, args, output_dir, device)
        
        fold_accs.append(best_acc)
        fold_f1s.append(best_f1)

    # --- 最終統計報告 ---
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    print("\n" + "="*50)
    print(f"   {args.k_folds}-Fold Cross Validation Summary")
    print("="*50)
    print(f"{'Fold':<6} | {'Accuracy':<12} | {'F1 Score (Macro)':<12}")
    print("-" * 40)
    for i in range(len(fold_accs)):
        print(f"{i:<6} | {fold_accs[i]:.2%}       | {fold_f1s[i]:.2%}")
    print("-" * 40)
    print(f"AVG    | {mean_acc:.2%} (+/-{std_acc:.2%}) | {mean_f1:.2%} (+/-{std_f1:.2%})")
    print("="*50)

    # 儲存摘要
    with open(os.path.join(output_dir, 'final_summary.txt'), 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"K-Folds: {args.k_folds}\n")
        f.write(f"Average Accuracy: {mean_acc:.2%} (+/- {std_acc:.2%})\n")
        f.write(f"Average F1 Score: {mean_f1:.2%} (+/- {std_f1:.2%})\n\n")
        f.write("Details per fold:\n")
        for i in range(len(fold_accs)):
            f.write(f"Fold {i}: Acc={fold_accs[i]:.4f}, F1={fold_f1s[i]:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['baseline', 'late_fusion', 'partition_fusion', 'partition_fusion_conv', 'partition_fusion_attn'])
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--fusion_features', type=str, default='both', choices=['first', 'second', 'both'])
    parser.add_argument(
        '--partition_hand_mode',
        type=str,
        default='both',
        choices=['both', 'none', 'left', 'right'],
        help="Partition 手部特徵選擇: both | none | left | right (外側手請選對應的左右)",
    )
    parser.add_argument('--use_class_weights', default=True, action='store_true')
    parser.add_argument('--no_class_weights', action='store_true')
    
    args = parser.parse_args()
    if args.no_class_weights:
        args.use_class_weights = False
    main(args)
