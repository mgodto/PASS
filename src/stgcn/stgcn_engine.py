import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def get_formatted_report_string(y_true, y_pred, class_names):
    """
    輔助函數：將分類報告轉換為百分比格式 (xx.xx%) 的字串表格
    """
    # 1. 獲取原始字典
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    # 2. 提取 accuracy
    accuracy_score_val = report_dict.pop('accuracy')
    
    # 3. 轉為 DataFrame
    df = pd.DataFrame(report_dict).transpose()
    
    # 4. 格式化函數 (轉為 %)
    def fmt_pct(x): return f"{x:.2%}"
    
    # 5. 將數值欄位轉為百分比
    for col in ['precision', 'recall', 'f1-score']:
        df[col] = df[col].apply(fmt_pct)
    
    # 6. Support 轉為整數字串
    df['support'] = df['support'].apply(lambda x: str(int(x)))
    
    # 7. 手動構建 Accuracy 行
    # 為了讓它出現在表格中間或最後，我們手動建立這一行
    total_support = df.loc['macro avg', 'support']
    accuracy_row = pd.DataFrame({
        'precision': [''],   
        'recall': [''],      
        'f1-score': [fmt_pct(accuracy_score_val)],
        'support': [total_support]
    }, index=['accuracy'])
    
    # 8. 重新排序：類別 -> Accuracy -> Macro/Weighted Avg
    avgs = df.loc[['macro avg', 'weighted avg']]
    classes = df.drop(['macro avg', 'weighted avg'])
    
    final_df = pd.concat([classes, accuracy_row, avgs])
    
    return final_df.to_string()

def train_one_epoch(model, data_loader, loss_fn, optimizer, device, model_type):
    """
    執行一個世代的訓練。
    """
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    processed_batches = 0
    
    for data in tqdm(data_loader, desc="Training"):
        # 解包邏輯
        if model_type == 'late_fusion' or model_type == 'partition_fusion':
            skeletons, subspace_features, labels = data
            skeletons = skeletons.to(device)
            subspace_features = subspace_features.to(device)
            labels = labels.to(device)
        else: 
            skeletons, labels = data
            skeletons = skeletons.to(device)
            labels = labels.to(device)

        valid_mask = labels >= 0
        if not valid_mask.all().item():
            valid_count = int(valid_mask.sum().item())
            if valid_count == 0:
                continue
            skeletons = skeletons[valid_mask]
            labels = labels[valid_mask]
            if model_type == 'late_fusion' or model_type == 'partition_fusion':
                subspace_features = subspace_features[valid_mask]
        
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == 'late_fusion' or model_type == 'partition_fusion':
            outputs = model(skeletons, subspace_features)
        else:
            outputs = model(skeletons)
            
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 統計
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        processed_batches += 1
        
    if processed_batches == 0 or total_samples == 0:
        return 0.0, 0.0
    avg_loss = total_loss / processed_batches
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def evaluate(model, data_loader, loss_fn, device, label_encoder, model_type):
    """
    在驗證/測試集上評估模型。
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    processed_batches = 0
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            # 解包
            if model_type == 'late_fusion' or model_type == 'partition_fusion':
                skeletons, subspace_features, labels = data
                skeletons = skeletons.to(device)
                subspace_features = subspace_features.to(device)
                labels = labels.to(device)
            else:
                skeletons, labels = data
                skeletons = skeletons.to(device)
                labels = labels.to(device)

            valid_mask = labels >= 0
            if not valid_mask.all().item():
                valid_count = int(valid_mask.sum().item())
                if valid_count == 0:
                    continue
                skeletons = skeletons[valid_mask]
                labels = labels[valid_mask]
                if model_type == 'late_fusion' or model_type == 'partition_fusion':
                    subspace_features = subspace_features[valid_mask]

            # Forward
            if model_type == 'late_fusion' or model_type == 'partition_fusion':
                outputs = model(skeletons, subspace_features)
            else:
                outputs = model(skeletons)
                
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            processed_batches += 1
            
    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
    if len(all_labels) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No valid samples", ha='center', va='center')
        ax.axis('off')
        return avg_loss, 0.0, 0.0, "No valid samples", fig
    
    # 轉回字串標籤
    y_true_str = label_encoder.inverse_transform(all_labels)
    y_pred_str = label_encoder.inverse_transform(all_preds)
    class_names = label_encoder.classes_

    accuracy = accuracy_score(y_true_str, y_pred_str)
    f1 = f1_score(y_true_str, y_pred_str, average='macro')
    
    # ★★★ 修改點：使用自定義函數生成百分比格式的報告 ★★★
    report = get_formatted_report_string(y_true_str, y_pred_str, class_names)
    
    # 繪製混淆矩陣 (維持整數顯示)
    cm = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    
    return avg_loss, accuracy, f1, report, fig
