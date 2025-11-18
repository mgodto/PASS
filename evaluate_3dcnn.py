import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torchinfo import summary

# 從 src 導入必要的模組
from src.video_dataset import VideoDataset
from src.cnn.cnn_models import Simple3DCNN
from src.config import RAW_VIDEO_DIR # <-- 我們需要這個來 fit LabelEncoder

def get_label_from_filename(filename: str) -> str:
    """從檔名中識別並回傳疾病標籤 (與訓練時相同)"""
    possible_labels = ["ASD", "LCS", "DHS", "HipOA"]
    for label in possible_labels:
        if label in filename:
            return label
    return None

def evaluate(model, data_loader, device, label_encoder):
    """
    在測試集上評估一個已經訓練好的 3D-CNN 模型。
    (此函數與之前版本相同，無需修改)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in data_loader:
            videos, labels = videos.to(device), labels.to(device)
            
            outputs = model(videos)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 將數字標籤轉回原始字串標籤
    y_true_str = label_encoder.inverse_transform(all_labels)
    y_pred_str = label_encoder.inverse_transform(all_preds)
    class_names = label_encoder.classes_

    # 計算指標
    accuracy = accuracy_score(y_true_str, y_pred_str)
    
    # 生成報告字典並轉換為DataFrame
    report_dict = classification_report(y_true_str, y_pred_str, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # 格式化為百分比
    report_df['precision'] = report_df['precision'].apply(lambda x: f"{x:.2%}")
    report_df['recall'] = report_df['recall'].apply(lambda x: f"{x:.2%}")
    report_df['f1-score'] = report_df['f1-score'].apply(lambda x: f"{x:.2%}")
    report_df['support'] = report_df['support'].astype(int)

    # 繪製混淆矩陣
    cm = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix for 3D-CNN Model')
    
    return accuracy, report_df, fig

def main(args):
    # 設定隨機種子 (雖然這裡不用切分，但保持習慣是好的)
    torch.manual_seed(43)

    # 設定設備
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    # --- ★★★ 關鍵修改點 (1)：Fit LabelEncoder ★★★ ---
    # 1. 掃描"原始訓練資料夾"，目的是建立與訓練時一致的 LabelEncoder
    print(f"Fitting LabelEncoder based on original training data dir: {RAW_VIDEO_DIR}")
    all_train_video_paths = glob.glob(os.path.join(RAW_VIDEO_DIR, '*'))
    train_labels_str = []
    for path in all_train_video_paths:
        label = get_label_from_filename(os.path.basename(path))
        if label:
            train_labels_str.append(label)
    
    if not train_labels_str:
        raise ValueError(f"No valid labels found in RAW_VIDEO_DIR: {RAW_VIDEO_DIR}. Cannot fit LabelEncoder.")

    le = LabelEncoder()
    le.fit(train_labels_str) # <-- 關鍵：只 fit，不 transform
    num_classes = len(le.classes_)
    print(f"LabelEncoder fitted with {num_classes} classes: {le.classes_}")
    
    # --- ★★★ 關鍵修改點 (2)：載入獨立的測試集 ★★★ ---
    # 2. 掃描您指定的"新測試集資料夾"
    print(f"Loading independent test set from: {args.test_dir}")
    test_video_paths_scan = glob.glob(os.path.join(args.test_dir, '*'))
    test_file_paths = []
    test_labels_str = []
    for path in test_video_paths_scan:
        label = get_label_from_filename(os.path.basename(path))
        if label:
            test_file_paths.append(path)
            test_labels_str.append(label)
            
    if not test_file_paths:
        raise ValueError(f"No video files with valid labels found in --test_dir: {args.test_dir}")

    # 3. 使用"已經 fit 好的" LabelEncoder 來轉換新測試集的標籤
    try:
        test_labels = le.transform(test_labels_str)
    except ValueError as e:
        print(f"Error: {e}. One or more labels in your test set were not present in the training data.")
        print("Labels found in test set:", set(test_labels_str))
        print("Labels known by LabelEncoder:", set(le.classes_))
        raise e

    # 4. 建立測試集 (不再需要 random_split)
    print(f"Creating test dataset with {len(test_file_paths)} videos.")
    # 注意：這裡的 VideoDataset 需要傳入 is_train=False (如果您的 VideoDataset 有實作數據增強的切換)
    # 根據您提供的 video_dataset.py，目前沒有這個參數，所以直接使用即可
    test_dataset = VideoDataset(test_file_paths, test_labels, num_frames=args.num_frames)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 5. 建立模型架構
    print("Building 3D-CNN model architecture...")
    model = Simple3DCNN(num_classes=num_classes).to(device)

    # 6. 載入已訓練好的模型權重
    print(f"Loading trained weights from: {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))

    # 7. 執行評估
    accuracy, report_df, cm_fig = evaluate(model, test_loader, device, le)
    
    # 8. 顯示與儲存結果 (與之前相同)
    print("\n" + "="*50)
    print("      Final 3D-CNN Model Evaluation on Test Set")
    print("="*50)
    print(f"\nModel Weights: {args.weights}")
    print(f"Test Set Path: {args.test_dir}")
    print(f"\nAccuracy on Test Set: {accuracy:.2%}")
    print("\nDetailed Classification Report:\n")
    print(report_df)
    print("="*50)
    
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    report_save_path = os.path.join(output_dir, f'3dcnn_report_{os.path.basename(args.test_dir)}.csv')
    report_df.to_csv(report_save_path)
    print(f"Classification report saved to: {report_save_path}")

    cm_save_path = os.path.join(output_dir, f'3dcnn_cm_{os.path.basename(args.test_dir)}.png')
    cm_fig.savefig(cm_save_path)
    print(f"Confusion matrix saved to: {cm_save_path}")
    
    dummy_input = torch.randn(1, 3, args.num_frames, 112, 112).to(device)
    print("\nModel Summary:")
    summary(model, input_data=dummy_input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained 3D-CNN model on a specific test set.')
    
    # --- ★★★ 關鍵修改點 (3)：修改 argparse ★★★ ---
    parser.add_argument('--weights', type=str, required=True, 
                        help='Path to the trained best_model.pth file.')
    parser.add_argument('--test_dir', type=str, required=True, 
                        help='Path to the folder containing the independent test set videos.')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for evaluation.')
    parser.add_argument('--num_frames', type=int, default=16, 
                        help='Number of frames used during training (must match).')
    
    args = parser.parse_args()
    main(args)