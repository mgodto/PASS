# import torch
# import argparse
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
# from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from torchinfo import summary

# # 從 src 導入必要的模組
# from src.stgcn_dataset import GaitDataset
# from src.stgcn_models import STGCN_Baseline, STGCN_LateFusion
# from src.config import SVM_FEATURES_PATH, LABELS_PATH, STGCN_PATHS_PATH

# def evaluate_model(model, data_loader, device, label_encoder, model_type):
#     """
#     在測試集上評估一個已經訓練好的模型。
#     """
#     model.eval()
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for data in data_loader:
#             if model_type == 'late_fusion':
#                 skeletons, subspace_features, labels = data
#                 skeletons, subspace_features, labels = skeletons.to(device), subspace_features.to(device), labels.to(device)
#             else: # baseline 模式
#                 skeletons, labels = data
#                 skeletons, labels = skeletons.to(device), labels.to(device)

#             if model_type == 'late_fusion':
#                 outputs = model(skeletons, subspace_features)
#             else:
#                 outputs = model(skeletons)
            
#             _, predicted = torch.max(outputs.data, 1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
            
#     # 將數字標籤轉回原始字串標籤
#     y_true_str = label_encoder.inverse_transform(all_labels)
#     y_pred_str = label_encoder.inverse_transform(all_preds)
#     class_names = label_encoder.classes_

#     # 計算指標
#     accuracy = accuracy_score(y_true_str, y_pred_str)
    
#     # 生成報告字典並轉換為DataFrame
#     report_dict = classification_report(y_true_str, y_pred_str, output_dict=True, zero_division=0)
#     report_df = pd.DataFrame(report_dict).transpose()
    
#     # 格式化為百分比
#     report_df['precision'] = report_df['precision'].apply(lambda x: f"{x:.2%}")
#     report_df['recall'] = report_df['recall'].apply(lambda x: f"{x:.2%}")
#     report_df['f1-score'] = report_df['f1-score'].apply(lambda x: f"{x:.2%}")
#     report_df['support'] = report_df['support'].astype(int)

#     # 繪製混淆矩陣
#     cm = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.ylabel('Actual Class')
#     plt.xlabel('Predicted Class')
#     plt.title(f'Confusion Matrix for {model_type} model')
    
#     return accuracy, report_df, fig

# def main(args):
#     # 設定隨機種子以確保數據分割的一致性
#     torch.manual_seed(42)

#     # 設定設備
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#         print("Using device: mps (Apple Silicon GPU)")
#     else:
#         device = torch.device("cpu")
#         print("Using device: cpu")

#     # 準備數據集 (與訓練時使用完全相同的流程和種子)
#     print("Loading and splitting dataset...")
#     full_dataset = GaitDataset(
#         stgcn_paths_file=STGCN_PATHS_PATH,
#         labels_file=LABELS_PATH,
#         subspace_features_file=SVM_FEATURES_PATH,
#         mode=args.model,
#         max_len=300 # 假設與訓練時相同
#     )
    
#     train_size = int(0.8 * len(full_dataset))
#     test_size = len(full_dataset) - train_size
#     _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
#     # 建立模型架構
#     print(f"Building model architecture for '{args.model}'...")
#     num_subspace_features = full_dataset.subspace_features.shape[1] if full_dataset.subspace_features is not None else 0
#     if args.model == 'baseline':
#         model = STGCN_Baseline(num_classes=full_dataset.num_classes).to(device)
#     elif args.model == 'late_fusion':
#         model = STGCN_LateFusion(
#             num_subspace_features=num_subspace_features,
#             num_classes=full_dataset.num_classes
#         ).to(device)
#     else:
#         raise ValueError("Invalid model type specified.")

#     # 載入已訓練好的模型權重
#     print(f"Loading trained weights from: {args.weights}")
#     ### weights_only=True 參數用於忽略不匹配的鍵 
#     model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))

#     # 執行評估
#     accuracy, report_df, cm_fig = evaluate_model(model, test_loader, device, full_dataset.le, args.model)
    
#     # 顯示結果
#     print("\n" + "="*45)
#     print("           Final Model Evaluation on Test Set")
#     print("="*45)
#     print(f"\nModel: {args.model}")
#     print(f"Weights: {args.weights}")
#     print(f"\nAccuracy on Test Set: {accuracy:.2%}")
#     print("\nDetailed Classification Report:\n")
#     print(report_df)
#     print("="*45)
    
#     # 儲存混淆矩陣圖
#     output_dir = os.path.dirname(args.weights)
#     report_filename = f'final_evaluation_report_{os.path.basename(args.weights)}.csv'
#     report_save_path = os.path.join(output_dir, report_filename)
#     report_df.to_csv(report_save_path)
#     print(f"Classification report saved to: {report_save_path}")

#     cm_filename = f'final_evaluation_cm_{os.path.basename(args.weights)}.png'
#     cm_save_path = os.path.join(output_dir, cm_filename)
#     cm_fig.savefig(cm_save_path)
#     print(f"Confusion matrix saved to: {cm_save_path}")
    
#     if args.model == 'baseline':
#         dummy_skeleton_input = torch.randn(1,3, 33, 200).to(device)
#         print("\nModel Summary:")
#         summary(model, input_data=dummy_skeleton_input)
#     else:
#         dummy_skeleton_input = torch.randn(1,3, 33, 200).to(device)
#         dummy_subspace_input = torch.randn(1, num_subspace_features).to(device) if num_subspace_features > 0 else None
#         print("\nModel Summary:")
#         summary(model, input_data=(dummy_skeleton_input, dummy_subspace_input))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Evaluate a trained ST-GCN model on the test set.')
#     parser.add_argument('--model', type=str, required=True, choices=['baseline', 'late_fusion'], help='Model architecture type.')
#     parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights file (.pth).')
#     parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
#     args = parser.parse_args()
#     main(args)



import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split # <-- 確保導入 random_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torchinfo import summary # 暫時移除 summary，因為它可能與 MPS 不完全兼容

# 從 src 導入必要的模組
from src.stgcn_dataset import GaitDataset
from src.stgcn_models import STGCN_Baseline, STGCN_LateFusion
from src.config import SVM_FEATURES_PATH, LABELS_PATH, STGCN_PATHS_PATH

def evaluate_model(model, data_loader, device, label_encoder, model_type, fusion_features='both'):
    """
    在測試集上評估一個已經訓練好的模型。
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in data_loader:
            # --- ★★★ 核心修正點 1：根據 fusion_features 解包數據 ★★★ ---
            # DataLoader 回傳的數據結構與訓練時一致
            if model_type == 'late_fusion':
                skeletons, subspace_features, labels = data
                skeletons, subspace_features, labels = skeletons.to(device), subspace_features.to(device), labels.to(device)
            else: # baseline 模式
                skeletons, labels = data
                skeletons, labels = skeletons.to(device), labels.to(device)

            # 模型呼叫方式已在 stgcn_engine.py 中修正，模型自己處理 edge_index
            if model_type == 'late_fusion':
                outputs = model(skeletons, subspace_features)
            else:
                outputs = model(skeletons)
            # --- 修正結束 ---
            
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
    # --- ★★★ 核心修正點 2：在標題中加入 fusion_features 資訊 ★★★ ---
    fusion_tag = f" (Fusion: {fusion_features})" if model_type == 'late_fusion' else ""
    plt.title(f'Confusion Matrix for {model_type}{fusion_tag} model')
    # --- 修正結束 ---
    
    return accuracy, report_df, fig

def main(args):
    # 設定隨機種子以確保數據分割的一致性
    torch.manual_seed(43)

    # 設定設備
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    # 準備數據集 (與訓練時使用完全相同的流程和種子)
    print(f"Loading and splitting dataset for model '{args.model}' with fusion features '{args.fusion_features}'...")
    try:
        # --- ★★★ 核心修正點 3：將 fusion_features 傳遞給 Dataset ★★★ ---
        full_dataset = GaitDataset(
            stgcn_paths_file=STGCN_PATHS_PATH,
            labels_file=LABELS_PATH,
            subspace_features_file=SVM_FEATURES_PATH,
            mode=args.model,
            max_len=300, # 假設與訓練時相同, 可以考慮也設為參數
            fusion_features=args.fusion_features # <-- 傳遞參數
        )
    except FileNotFoundError as e:
         print(f"Error initializing dataset: {e}")
         return
    except Exception as e:
         print(f"An unexpected error occurred during dataset initialization: {e}")
         return
    # --- 修正結束 ---
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    # 使用 random_split 確保得到與訓練時相同的測試集
    _, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 建立模型架構
    print(f"Building model architecture for '{args.model}'...")
    # --- ★★★ 核心修正點 4：從 Dataset 獲取正確的特徵數量 ★★★ ---
    num_selected_subspace_features = full_dataset.num_selected_subspace_features
    print(f"Number of subspace features passed to the model: {num_selected_subspace_features}")
    # --- 修正結束 ---

    if args.model == 'baseline':
        model = STGCN_Baseline(num_classes=full_dataset.num_classes).to(device)
    elif args.model == 'late_fusion':
        model = STGCN_LateFusion(
            num_subspace_features=num_selected_subspace_features, # <-- 傳遞正確數量
            num_classes=full_dataset.num_classes
        ).to(device)
    else:
        raise ValueError("Invalid model type specified.")

    # 載入已訓練好的模型權重
    if not os.path.exists(args.weights):
        print(f"錯誤：找不到指定的權重檔案： {args.weights}")
        return
        
    print(f"Loading trained weights from: {args.weights}")
    try:
        # 使用 map_location=device 確保權重被載入到正確的設備
        # weights_only=True 是 PyTorch 1.13+ 的推薦選項，更安全
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    except Exception as e:
        print(f"載入權重時發生錯誤：{e}")
        print("請確認權重檔案與模型架構 (--model) 是否匹配。")
        return

    # 執行評估
    accuracy, report_df, cm_fig = evaluate_model(model, test_loader, device, full_dataset.le, args.model, args.fusion_features)
    
    # 顯示結果
    print("\n" + "="*45)
    print("           Final Model Evaluation on Test Set")
    print("="*45)
    print(f"\nModel: {args.model}")
    # --- ★★★ 核心修正點 5：顯示 fusion_features 資訊 ★★★ ---
    if args.model == 'late_fusion':
        print(f"Fusion Features: {args.fusion_features}")
    # --- 修正結束 ---
    print(f"Weights: {args.weights}")
    print(f"\nAccuracy on Test Set: {accuracy:.2%}")
    print("\nDetailed Classification Report:\n")
    print(report_df)
    print("="*45)
    
    # 儲存結果到模型權重所在的資料夾
    output_dir = os.path.dirname(args.weights) if os.path.dirname(args.weights) else '.' # 如果權重在根目錄
    
    # --- ★★★ 核心修正點 6：在檔名中加入 fusion_features 資訊 ★★★ ---
    base_weight_name = os.path.splitext(os.path.basename(args.weights))[0]
    fusion_tag_file = f"_fusion-{args.fusion_features}" if args.model == 'late_fusion' else ""
    
    report_filename = f'final_evaluation_report_{base_weight_name}{fusion_tag_file}.csv'
    report_save_path = os.path.join(output_dir, report_filename)
    try:
        report_df.to_csv(report_save_path)
        print(f"Classification report saved to: {report_save_path}")
    except Exception as e:
        print(f"儲存報告時發生錯誤：{e}")

    cm_filename = f'final_evaluation_cm_{base_weight_name}{fusion_tag_file}.png'
    cm_save_path = os.path.join(output_dir, cm_filename)
    try:
        cm_fig.savefig(cm_save_path)
        print(f"Confusion matrix saved to: {cm_save_path}")
        plt.close(cm_fig) # 關閉圖表釋放記憶體
    except Exception as e:
        print(f"儲存混淆矩陣時發生錯誤：{e}")
        plt.close(cm_fig) # 即使儲存失敗也要關閉

    # 暫時移除 torchinfo summary，因為它可能與 MPS 不完全兼容
    print("\nModel Summary:")
    try:
        if args.model == 'baseline':
            # 需要一個範例輸入來推斷形狀
            # 注意：這裡的 T (時間步) 可能需要根據實際情況調整
            summary(model, input_size=(args.batch_size, 3, 300, 33))
        else:
            # Late Fusion 的輸入是兩個張量
            summary(model, input_size=[(args.batch_size, 3, 300, 33), (args.batch_size, num_selected_subspace_features)])
    except Exception as e:
        print(f"無法生成模型摘要：{e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained ST-GCN model on the test set.')
    parser.add_argument('--model', type=str, required=True, choices=['baseline', 'late_fusion'], help='Model architecture type.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights file (.pth).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
    # --- ★★★ 核心修正點 7：加入 fusion_features 參數 ★★★ ---
    parser.add_argument('--fusion_features', type=str, default='both', choices=['first', 'second', 'both'],
                        help="Which subspace features were used for the late_fusion model ('first', 'second', or 'both')")
    # --- 修正結束 ---
    args = parser.parse_args()
    main(args)