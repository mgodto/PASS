#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stgcn_evaluate_fixed.py
- Evaluates ST-GCN Baseline / Late-Fusion with *exactly the same* split and class order as training.
- Removes random re-splitting during evaluation.
- Loads saved indices (results/train_indices.npy, results/test_indices.npy) and class order (results/classes.npy).
- Falls back gracefully with a clear error message if required files are missing.
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Project-local imports (assumes your package/module layout)
try:
    from src.stgcn_dataset import GaitDataset
    from src.stgcn_models import STGCN_Baseline, STGCN_LateFusion
    from src.config import SVM_FEATURES_PATH, LABELS_PATH, STGCN_PATHS_PATH
except Exception as e:
    print("❌ Import error. Make sure you can import from src/ (PYTHONPATH).")
    print(e)
    sys.exit(1)


def set_device():
    if torch.backends.mps.is_available():
        print("Using device: mps (Apple Silicon GPU)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using device: cuda")
        return torch.device("cuda")
    print("Using device: cpu")
    return torch.device("cpu")


def load_splits_or_die(results_dir="results"):
    train_idx_path = os.path.join(results_dir, "train_indices.npy")
    test_idx_path  = os.path.join(results_dir, "test_indices.npy")
    if not (os.path.exists(train_idx_path) and os.path.exists(test_idx_path)):
        raise FileNotFoundError(
            "Missing split indices.\n"
            f"  Expected:\n  - {train_idx_path}\n  - {test_idx_path}\n"
            "Please save them during training (see patch instructions), then re-run evaluation."
        )
    train_idx = np.load(train_idx_path, allow_pickle=True)
    test_idx  = np.load(test_idx_path, allow_pickle=True)
    return train_idx, test_idx


def load_classes_or_warn(results_dir="results"):
    classes_path = os.path.join(results_dir, "classes.npy")
    if os.path.exists(classes_path):
        classes = np.load(classes_path, allow_pickle=True).tolist()
        print(f"Loaded fixed class order from {classes_path}: {classes}")
        return classes
    print("⚠️  classes.npy not found. Will compute metrics by numeric indices only.")
    return None


def build_datasets(args, classes=None):
    # Always create the *full* dataset deterministically, then select subsets by saved indices.
    full = GaitDataset(
        stgcn_paths_file=STGCN_PATHS_PATH,
        labels_file=LABELS_PATH,
        subspace_features_file=SVM_FEATURES_PATH,
        mode='late_fusion' if args.model == 'late_fusion' else 'baseline',
        max_len=300
    )

    # If classes were saved during training, force the same order on the dataset encoder if possible.
    if classes is not None and hasattr(full, "le"):
        try:
            # Refit encoder on the fixed order so inverse_transform / reports use the same mapping
            import numpy as _np
            from sklearn.preprocessing import LabelEncoder as _LE
            le = _LE()
            le.fit(classes)
            full.le = le
        except Exception as e:
            print(f"⚠️  Could not rebind dataset label encoder to fixed classes: {e}")

    # Enforce fusion_features to BOTH when evaluating a late-fusion model (to match training setup)
    if args.model == 'late_fusion' and hasattr(full, "set_fusion_features"):
        try:
            full.set_fusion_features('both')
            print("Late-fusion evaluation set to use BOTH subspace feature groups.")
        except Exception as e:
            print(f"⚠️  Could not force fusion_features='both': {e}")

    # Load saved indices
    train_idx, test_idx = load_splits_or_die(args.results_dir)
    test_dataset = Subset(full, test_idx)
    return test_dataset, getattr(full, 'num_selected_subspace_features', None), classes


def build_model(args, device, num_subspace_features=None, num_classes=None):
    if args.model == 'baseline':
        model = STGCN_Baseline(num_classes=num_classes if num_classes else 4)  # fallback 4
    else:
        if num_subspace_features is None:
            raise ValueError("num_subspace_features is required for late_fusion.")
        model = STGCN_LateFusion(
            num_classes=num_classes if num_classes else 4,
            num_subspace_features=int(num_subspace_features)
        )
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_eval(model, loader, device, classes=None, model_type='baseline'):
    all_preds, all_labels = [], []
    for batch in loader:
        if model_type == 'late_fusion':
            skel, subspace, y = batch
            skel, subspace, y = skel.to(device), subspace.to(device), y.to(device)
            logits = model(skel, subspace)
        else:
            skel, y = batch
            skel, y = skel.to(device), y.to(device)
            logits = model(skel)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    # Confusion matrix and report
    labels_for_report = None
    target_names = None
    if classes is not None:
        labels_for_report = list(range(len(classes)))
        target_names = classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels_for_report, target_names=target_names, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=labels_for_report)
    return acc, cm, target_names


def plot_confusion(cm, class_names=None, out_png="results/confusion_matrix_eval.png"):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto",
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Eval)")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['baseline', 'late_fusion'])
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--results_dir', type=str, default='results', help='Where train_indices.npy/test_indices.npy/classes.npy are stored')
    args = parser.parse_args()

    device = set_device()

    classes = load_classes_or_warn(args.results_dir)
    test_dataset, num_subspace_features, classes = build_datasets(args, classes=classes)

    if classes is not None:
        num_classes = len(classes)
    else:
        # fall back to dataset attribute if available
        try:
            num_classes = len(test_dataset.dataset.le.classes_)
        except Exception:
            num_classes = 4

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args, device, num_subspace_features=num_subspace_features, num_classes=num_classes)

    acc, cm, names = run_eval(model, test_loader, device, classes=classes, model_type=args.model)
    plot_confusion(cm, class_names=names)

if __name__ == '__main__':
    main()
