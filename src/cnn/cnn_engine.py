import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for videos, labels in tqdm(data_loader, desc="Training"):
        videos, labels = videos.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate(model, data_loader, loss_fn, device, label_encoder):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(data_loader, desc="Evaluating"):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    
    y_true_str = label_encoder.inverse_transform(all_labels)
    y_pred_str = label_encoder.inverse_transform(all_preds)
    class_names = label_encoder.classes_

    accuracy = accuracy_score(y_true_str, y_pred_str)
    report = classification_report(y_true_str, y_pred_str, zero_division=0)
    
    cm = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    
    return avg_loss, accuracy, report, fig
