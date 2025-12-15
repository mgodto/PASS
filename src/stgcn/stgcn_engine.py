import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_one_epoch(model, data_loader, loss_fn, optimizer, device, model_type):
    """
    執行一個世代的訓練。
    """
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for data in tqdm(data_loader, desc="Training"):
        # ★★★ 修正點 1：將 partition_fusion 加入判斷條件 ★★★
        if model_type == 'late_fusion' or model_type == 'partition_fusion':
            # 解包三個變數
            skeletons, subspace_features, labels = data
            skeletons = skeletons.to(device)
            subspace_features = subspace_features.to(device)
            labels = labels.to(device)
        else: # baseline 模式
            skeletons, labels = data
            skeletons = skeletons.to(device)
            labels = labels.to(device)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # ★★★ 修正點 2：根據模式傳入正確參數 ★★★
        if model_type == 'late_fusion' or model_type == 'partition_fusion':
            # 這兩個模式都需要兩個輸入
            outputs = model(skeletons, subspace_features)
        else: # baseline 模式
            outputs = model(skeletons)
            
        # 計算損失
        loss = loss_fn(outputs, labels)
        
        # 反向傳播與優化
        loss.backward()
        optimizer.step()
        
        # 統計數據
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
    avg_loss = total_loss / len(data_loader)
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
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            # ★★★ 修正點 3：Evaluation 迴圈也要同步修正 ★★★
            if model_type == 'late_fusion' or model_type == 'partition_fusion':
                skeletons, subspace_features, labels = data
                skeletons = skeletons.to(device)
                subspace_features = subspace_features.to(device)
                labels = labels.to(device)
            else: # baseline 模式
                skeletons, labels = data
                skeletons = skeletons.to(device)
                labels = labels.to(device)

            # ★★★ 修正點 4：傳入參數 ★★★
            if model_type == 'late_fusion' or model_type == 'partition_fusion':
                outputs = model(skeletons, subspace_features)
            else: # baseline 模式
                outputs = model(skeletons)
                
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    
    # 將數字標籤轉回原始字串標籤
    y_true_str = label_encoder.inverse_transform(all_labels)
    y_pred_str = label_encoder.inverse_transform(all_preds)
    class_names = label_encoder.classes_

    accuracy = accuracy_score(y_true_str, y_pred_str)
    report = classification_report(y_true_str, y_pred_str, zero_division=0)
    
    # 繪製混淆矩陣
    cm = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    
    return avg_loss, accuracy, report, fig