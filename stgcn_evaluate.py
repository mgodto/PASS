import torch
import argparse
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# 從 src 導入必要的模組
from src.stgcn.stgcn_dataset import GaitDataset
from src.stgcn.stgcn_models import STGCN_Baseline, STGCN_LateFusion, STGCN_PartitionFusion
from src.stgcn.stgcn_metrics import (
    format_classification_report_percent,
    plot_confusion_matrix,
)
from src.config import SVM_FEATURES_PATH, LABELS_PATH, STGCN_PATHS_PATH, PARTITION_NPY_DIR

def get_formatted_classification_report(y_true, y_pred, class_names):
    """
    生成百分比格式的 Classification Report (文字表格)
    """
    return format_classification_report_percent(y_true, y_pred, class_names)

def evaluate_model(model, data_loader, device, label_encoder, model_type):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            # 解包數據
            if model_type == 'late_fusion' or model_type == 'partition_fusion':
                skeletons, subspace_features, labels = data
                skeletons = skeletons.to(device)
                subspace_features = subspace_features.to(device)
                labels = labels.to(device)
            else: # baseline
                skeletons, labels = data
                skeletons = skeletons.to(device)
                labels = labels.to(device)

            # 模型推論
            if model_type == 'late_fusion' or model_type == 'partition_fusion':
                outputs = model(skeletons, subspace_features)
            else:
                outputs = model(skeletons)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 轉換標籤
    y_true_str = label_encoder.inverse_transform(all_labels)
    y_pred_str = label_encoder.inverse_transform(all_preds)
    class_names = label_encoder.classes_

    # 1. 獲取百分比格式的報告
    report_str, accuracy = get_formatted_classification_report(y_true_str, y_pred_str, class_names)
    
    # 2. 繪製混淆矩陣 (改回顯示整數數量)
    fig = plot_confusion_matrix(
        y_true_str,
        y_pred_str,
        class_names,
        title=f'Confusion Matrix ({model_type}) - Counts',
    )
    
    return accuracy, report_str, fig, len(all_labels)

def main(args):
    #####################################
    torch.manual_seed(43)        ########
    #####################################
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    # 準備 Dataset
    print(f"Loading dataset info for '{args.model}' mode...")
    try:
        full_dataset = GaitDataset(
            stgcn_paths_file=STGCN_PATHS_PATH,
            labels_file=LABELS_PATH,
            subspace_features_file=SVM_FEATURES_PATH,
            mode=args.model,
            max_len=args.max_len,
            fusion_features=args.fusion_features,
            partition_features_dir=PARTITION_NPY_DIR
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return


    # 還原 Train/Test Split (8:2)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Evaluated Samples: {len(test_dataset)} (Matches validation set size)")

    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 建立模型
    num_classes = full_dataset.num_classes
    num_feats = full_dataset.num_selected_subspace_features

    if args.model == 'baseline':
        model = STGCN_Baseline(num_classes=num_classes).to(device)
    elif args.model == 'late_fusion':
        model = STGCN_LateFusion(num_subspace_features=num_feats, num_classes=num_classes).to(device)
    elif args.model == 'partition_fusion':
        print(f"Initializing STGCN_PartitionFusion with {num_feats} features...")
        model = STGCN_PartitionFusion(num_classes=num_classes, subspace_dim=num_feats).to(device)
        
    # 載入權重
    print(f"Loading weights from: {args.weights}")
    if os.path.exists(args.weights):
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Weights file not found.")
        return

    # 執行評估
    accuracy, report_str, cm_fig, num_samples = evaluate_model(model, data_loader, device, full_dataset.le, args.model)
    
    print("\n" + "="*60)
    print(f"Evaluation Results ({args.model})")
    print("="*60)
    print(f"Samples Evaluated: {num_samples}")
    print(f"Accuracy on Test Set:  {accuracy:.2%}")
    print("\nDetailed Classification Report:")
    #print("-" * 60)
    print(report_str) 
    print("-" * 60)
    
    # 儲存
    output_dir = os.path.dirname(args.weights)
    if not output_dir: output_dir = "."
        
    report_save_path = os.path.join(output_dir, 'final_evaluation_report.txt')
    cm_save_path = os.path.join(output_dir, 'final_evaluation_cm.png')
    
    with open(report_save_path, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Evaluated Samples: {num_samples}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n\n")
        f.write(report_str)
    print(f"Report saved to: {report_save_path}")
    
    cm_fig.savefig(cm_save_path)
    print(f"Confusion matrix saved to: {cm_save_path}")
    plt.close(cm_fig)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['baseline', 'late_fusion', 'partition_fusion'])
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--fusion_features', type=str, default='both', choices=['first', 'second', 'both'])

    args = parser.parse_args()
    main(args)
