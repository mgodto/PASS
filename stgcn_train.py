# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import numpy as np
# import os
# import argparse
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import time
# import random
# from datetime import datetime

# from src.stgcn_dataset import GaitDataset
# from src.stgcn_models import STGCN_Baseline, STGCN_LateFusion
# from src.stgcn_engine import train_one_epoch, evaluate
# from src.config import SVM_FEATURES_PATH, LABELS_PATH, STGCN_PATHS_PATH

# from src.loss import FocalLoss # ★★★ 測試看看導入 FocalLoss 取代原本的cross entropy loss ★★★

# from sklearn.utils.class_weight import compute_class_weight
# from torch.utils.data import WeightedRandomSampler


# def set_seed(seed):
#     """設定隨機數種子以確保可重複性"""
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # 如果使用 CUDA
#     if torch.backends.mps.is_available(): # 如果使用 MPS
#         torch.mps.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def main(args):
#     # ★★★ 在所有操作前設定隨機數種子 ★★★
#     # set_seed(42) 
#     # 檢查是否有可用的 Apple Silicon GPU (MPS)
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#         print("Using device: mps (Apple Silicon GPU)")
#     else:
#         device = torch.device("cpu")
#         print("Using device: cpu")



#     # 數據集準備
#     print(f"Loading dataset in '{args.model}' mode...")
#     dataset = GaitDataset(
#         stgcn_paths_file=STGCN_PATHS_PATH,
#         labels_file=LABELS_PATH,
#         subspace_features_file=SVM_FEATURES_PATH,
#         mode=args.model,
#         max_len=args.max_len
#     )

#     ########################################################################################
#     # 計算類別權重以處理不平衡數據集 ############################################################
#     class_weights = compute_class_weight(
#     'balanced',
#     classes=np.unique(dataset.labels),
#     y=dataset.labels
#     )
#     class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

#     print("Calculated Class Weights:")
#     for i, weight in enumerate(class_weights):
#         class_name = dataset.le.inverse_transform([i])[0]
#         print(f"- {class_name}: {weight:.2f}")
#     #####################################################################

#     # 數據加載器
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

#     ##################################################################################################
#     # ★★★ 新增：為訓練集建立加權採樣器 ★★★ ###############################################################
#     # 1. 獲取訓練集中所有樣本的標籤
#     # train_labels = [train_dataset.dataset.labels[i] for i in train_dataset.indices]

#     # # 2. 計算每個類別的權重 (與方法一類似，但用於採樣)
#     # class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)

#     # # 3. 為訓練集中的每個樣本分配一個權重
#     # sample_weights = [class_weights[label] for label in train_labels]

#     # # 4. 建立採樣器
#     # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
#     ####################################################################################################

#     # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
#     # 模型建立
#     print("Building model...")
#     # 從數據集中安全地獲取部分空間法特徵的數量
#     num_subspace_features = 0
#     if args.model == 'late_fusion' and dataset.subspace_features is not None:
#         num_subspace_features = dataset.subspace_features.shape[1]
#         print(f"Detected {num_subspace_features} subspace features for late fusion.")
#     num_subspace_features = dataset.subspace_features.shape[1] if dataset.subspace_features is not None else 0

#     if args.model == 'baseline':
#         model = STGCN_Baseline(num_classes=dataset.num_classes).to(device)
#     elif args.model == 'late_fusion':
#         model = STGCN_LateFusion(
#             num_subspace_features=num_subspace_features,
#             num_classes=dataset.num_classes
#         ).to(device)
#     else:
#         raise ValueError("Invalid model type specified.")

#     # 損失函數與優化器
#     class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

#     # loss function 的選擇
#     # cross entropy loss
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#     # criterion = nn.CrossEntropyLoss()

#     # focal loss
#     # criterion = FocalLoss(alpha=class_weights, gamma=2.0)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

#     # 安排學習率調度器 (可選)
#     # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

#     # --- ★★★ 新增：建立歷史紀錄串列 ★★★ ---
#     history = {
#         'train_loss': [],
#         'train_acc': [],
#         'test_loss': [],
#         'test_acc': []
#     }

#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     experiment_name = f"STGCN_{args.model}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
#     output_dir = os.path.join('results', experiment_name)
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"本次實驗結果將儲存於: {output_dir}")
#     # ★★★ 2. 記錄訓練開始時間 ★★★
#     training_start_time = time.time()

#     # 訓練循環
#     best_accuracy = 0.0
#     for epoch in range(args.epochs):
#         print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
#         # --- ★★★ 這就是修正的部分 ★★★ ---
#         # 修正 train_one_epoch 和 evaluate 的參數順序
#         train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.model)
#         test_loss, test_acc, report, cm_fig = evaluate(model, test_loader, criterion, device, dataset.le, args.model)
#         # --- 修正結束 ---

#         # 若有使用scheduler，請不要註解掉這兩行
#         scheduler.step()
#         print(f"Current Learning Rate: {scheduler.get_last_lr()[0]}") # 顯示當前的學習率

#         # 記錄當前 epoch 的結果
#         print(f"Epoch {epoch+1} Results: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")
        

#         # --- ★★★ 新增：紀錄當前 epoch 的指標 ★★★ ---
#         history['train_loss'].append(train_loss)
#         history['train_acc'].append(train_acc)
#         history['test_loss'].append(test_loss)
#         history['test_acc'].append(test_acc)

#         # 儲存最佳模型
#         if test_acc > best_accuracy:
#             best_accuracy = test_acc
#             save_path = os.path.join(output_dir, 'best_model.pth')
#             torch.save(model.state_dict(), save_path)
#             print(f"New best model saved to {save_path} with accuracy: {best_accuracy:.2%}")
        
#         # 儲存每個世代的評估報告
#         report_save_path = os.path.join(output_dir, f'epoch_{epoch+1}_report.txt')
#         with open(report_save_path, 'w') as f:
#             f.write(report)
#         cm_save_path = os.path.join(output_dir, f'epoch_{epoch+1}_cm.png')
#         cm_fig.savefig(cm_save_path)
#         plt.close(cm_fig)


#     # ★★★ 3. 計算並顯示總訓練時間 ★★★
#     training_end_time = time.time()
#     total_training_time = training_end_time - training_start_time
    
#     # 將秒數格式化為 小時:分鐘:秒
#     hours = int(total_training_time // 3600)
#     minutes = int((total_training_time % 3600) // 60)
#     seconds = int(total_training_time % 60)

#     # --- ★★★ 新增：繪製並儲存準確率與損失變化圖 ★★★ ---
#     print(f"Total Training Time: {hours}h {minutes}m {seconds}s")
#     print("\nTraining finished. Plotting history...")
#     epochs_range = range(1, args.epochs + 1)
    
#     # 繪製準確率圖
#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
#     plt.plot(epochs_range, history['test_acc'], label='Testing Accuracy')
#     plt.title('Accuracy vs. Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(loc='lower right')
#     plt.grid(True)
#     accuracy_plot_path = f'results/accuracy_vs_epochs_{args.model}.png'
#     plt.savefig(accuracy_plot_path)
#     print(f"Accuracy plot saved to {accuracy_plot_path}")
#     plt.close()

#     # 繪製損失圖
#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs_range, history['train_loss'], label='Training Loss')
#     plt.plot(epochs_range, history['test_loss'], label='Testing Loss')
#     plt.title('Loss vs. Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend(loc='upper right')
#     plt.grid(True)
#     loss_plot_path = f'results/loss_vs_epochs_{args.model}.png'
#     plt.savefig(loss_plot_path)
#     print(f"Loss plot saved to {loss_plot_path}")
#     plt.close()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='ST-GCN Training Script for Gait Analysis')
#     parser.add_argument('--model', type=str, required=True, choices=['baseline', 'late_fusion'], help='Model type to train')
#     parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
#     parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
#     parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')  # originally 0.001
#     parser.add_argument('--fusion_features', type=str, default='both', choices=['first', 'second', 'both'], 
#                         help="Which subspace features to use in late_fusion mode ('first', 'second', or 'both')")
#     ## 0.001 stuck in local minimum (baseline)
#     ## 0.01 stuck in local minimum but better than 0.001 (baseline)
#     ## 0.05 get best result (baseline)

#     ## 0.05 very unstable(late_fusion)
#     ## 0.0001 loss有在下降但是太慢了(late_fusion)
#     ## 0.0005 使用scheduler 每15個epoch衰減0.5 78% 比baseline好 (late_fusion)
#     ## 0.0005 使用scheduler 每10個epoch衰減0.7 (late_fusion)
#     parser.add_argument('--max_len', type=int, default=300, help='Maximum sequence length for padding/truncating')
#     args = parser.parse_args()
#     main(args)


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
from src.stgcn.stgcn_models import STGCN_Baseline, STGCN_LateFusion
from src.stgcn.stgcn_engine import train_one_epoch, evaluate
from src.config import SVM_FEATURES_PATH, LABELS_PATH, STGCN_PATHS_PATH

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
            fusion_features=args.fusion_features # <-- 將參數傳遞給 Dataset
        )
    except FileNotFoundError as e:
         print(f"Error initializing dataset: {e}")
         return # 數據文件缺失則退出
    except Exception as e:
         print(f"An unexpected error occurred during dataset initialization: {e}")
         return

    # 數據分割
    #     ##################################################################################################
#     # ★★★ 新增：為訓練集建立加權採樣器 ★★★ ###############################################################
#     # 1. 獲取訓練集中所有樣本的標籤
#     # train_labels = [train_dataset.dataset.labels[i] for i in train_dataset.indices]

#     # # 2. 計算每個類別的權重 (與方法一類似，但用於採樣)
#     # class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)

#     # # 3. 為訓練集中的每個樣本分配一個權重
#     # sample_weights = [class_weights[label] for label in train_labels]

#     # # 4. 建立採樣器
#     # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
#     ####################################################################################################

#     # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)



    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    

    # ***************************** Fix: Save dataset splits and class order *****************************
    # --- Add right after you create your full dataset and split it ---
    # from pathlib import Path

    # results_dir = Path("results")
    # results_dir.mkdir(parents=True, exist_ok=True)

    # # If you used torch.utils.data.random_split -> you got Subset objects.
    # # Save their indices so evaluation can reuse the EXACT same split.
    # def _extract_indices(split_obj):
    #     # If using Subset, it has .indices
    #     if hasattr(split_obj, "indices"):
    #         return np.array(split_obj.indices)
    #     # Otherwise, assume it is a list/np.array already
    #     return np.asarray(split_obj, dtype=int)

    # np.save(results_dir / "train_indices.npy", _extract_indices(train_dataset))
    # np.save(results_dir / "test_indices.npy",  _extract_indices(test_dataset))

    # # Also save the class order so reports/labels match exactly.
    # try:
    #     classes = full_dataset.le.classes_.tolist()
    # except Exception:
    #     # Fallback to whatever you use to define your classes, e.g. dataset attribute
    #     classes = getattr(full_dataset, "classes", None)

    # if classes is not None:
    #     np.save(results_dir / "classes.npy", np.array(classes, dtype=object))
    #     print("Saved classes.npy:", classes)
    # else:
    #     print("⚠️ Could not save classes.npy (no class list found).")
    # # --- End of patch ---

    # # ***************************** Fix: Save dataset splits and class order *****************************



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
        test_loss, test_acc, report, cm_fig = evaluate(model, test_loader, criterion, device, dataset.le, args.model)
        
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
    parser.add_argument('--model', type=str, required=True, choices=['baseline', 'late_fusion'], help='Model type to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=300, help='Maximum sequence length for padding/truncating')
    parser.add_argument('--fusion_features', type=str, default='both', choices=['first', 'second', 'both'], 
                        help="Which subspace features to use in late_fusion mode ('first', 'second', or 'both')")
    # --- ★★★ 新增：控制是否使用類別權重的參數 ★★★ ---
    parser.add_argument('--use_class_weights', default=True, action='store_true', 
                        help='Apply class weighting to the loss function to handle imbalance.')
    # --- 新增結束 ---
    args = parser.parse_args()
    main(args)

