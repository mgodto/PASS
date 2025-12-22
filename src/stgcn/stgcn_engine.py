import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from src.stgcn.stgcn_metrics import (
    format_classification_report_percent,
    plot_confusion_matrix,
)


def _is_fusion_mode(model_type):
    return model_type in ('late_fusion', 'partition_fusion')


def _unpack_batch(data, device, model_type):
    if _is_fusion_mode(model_type):
        skeletons, subspace_features, labels = data
        return skeletons.to(device), subspace_features.to(device), labels.to(device)
    skeletons, labels = data
    return skeletons.to(device), None, labels.to(device)


def _filter_valid_samples(skeletons, subspace_features, labels, model_type):
    valid_mask = labels >= 0
    if valid_mask.all().item():
        return skeletons, subspace_features, labels
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return None, None, None
    skeletons = skeletons[valid_mask]
    labels = labels[valid_mask]
    if _is_fusion_mode(model_type) and subspace_features is not None:
        subspace_features = subspace_features[valid_mask]
    return skeletons, subspace_features, labels


def _forward_model(model, skeletons, subspace_features, model_type):
    if _is_fusion_mode(model_type):
        return model(skeletons, subspace_features)
    return model(skeletons)


def get_formatted_report_string(y_true, y_pred, class_names):
    """
    輔助函數：將分類報告轉換為百分比格式 (xx.xx%) 的字串表格
    """
    report_str, _ = format_classification_report_percent(y_true, y_pred, class_names)
    return report_str

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
        skeletons, subspace_features, labels = _unpack_batch(data, device, model_type)
        skeletons, subspace_features, labels = _filter_valid_samples(
            skeletons,
            subspace_features,
            labels,
            model_type,
        )
        if skeletons is None:
            continue
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = _forward_model(model, skeletons, subspace_features, model_type)
            
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
            skeletons, subspace_features, labels = _unpack_batch(data, device, model_type)
            skeletons, subspace_features, labels = _filter_valid_samples(
                skeletons,
                subspace_features,
                labels,
                model_type,
            )
            if skeletons is None:
                continue

            # Forward
            outputs = _forward_model(model, skeletons, subspace_features, model_type)
                
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
    fig = plot_confusion_matrix(
        y_true_str,
        y_pred_str,
        class_names,
        title='Confusion Matrix',
    )
    
    return avg_loss, accuracy, f1, report, fig
