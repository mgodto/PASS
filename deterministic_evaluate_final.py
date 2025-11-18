import os
import re
import glob
import json
import argparse
import random
from typing import List, Sequence

import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

# Prefer strict dataset (raises on unreadable videos); fallback to permissive one.
try:
    from video_dataset_strict import VideoDatasetStrict as VideoDataset
except ImportError:
    from src.video_dataset import VideoDataset

from src.cnn_models import Simple3DCNN


# ===== Determinism helpers =====
def set_deterministic(seed: int = 43):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===== File listing & label extraction =====
VALID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
def is_video(p): return os.path.splitext(p)[1].lower() in VALID_EXTS

def list_videos(folder: str):
    return sorted([p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if is_video(p)])

def extract_label_from_filename(fname: str, regex: str, tokens: List[str]):
    m = re.search(regex, fname, flags=re.IGNORECASE)
    if m:
        g = m.group(1) if m.groups() else m.group(0)
        return g.upper()
    upper = fname.upper()
    for t in tokens:
        if t.upper() in upper:
            return t.upper()
    return None


# ===== Evaluation　=====
def evaluate_like_before(model, data_loader, device, label_encoder):
    model.eval()
    all_preds, all_labels = [], []
    did_log = False
    with torch.no_grad():
        for videos, labels in data_loader:
            videos = videos.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if not did_log:
                try:
                    print("video batch device:", videos.device, "model device:", next(model.parameters()).device)
                except Exception:
                    pass
                did_log = True
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # map back to string labels
    y_true_str = label_encoder.inverse_transform(all_labels)
    y_pred_str = label_encoder.inverse_transform(all_preds)
    class_names = list(label_encoder.classes_)

    acc = accuracy_score(y_true_str, y_pred_str)

    report_dict = classification_report(y_true_str, y_pred_str, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    for col in ["precision", "recall", "f1-score"]:
        if col in report_df.columns:
            report_df[col] = report_df[col].apply(lambda x: f"{x:.2%}")
    if "support" in report_df.columns:
        report_df["support"] = report_df["support"].astype(int)

    # Confusion matrix fig
    cm = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("Actual Class"); ax.set_xlabel("Predicted Class")
    ax.set_title("Confusion Matrix for 3D-CNN Model")

    return acc, report_df, fig


# ===== Model utils =====
def infer_num_classes_from_state(state_dict):
    # try typical classifier names
    for key in ["classifier.4.weight", "classifier.3.weight", "fc.weight", "head.weight"]:
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return state_dict[key].shape[0]
    # fallback: pick the last 2D weight's out_features
    max_out = None
    for k, w in state_dict.items():
        if k.endswith(".weight") and getattr(w, "ndim", 0) == 2:
            out = w.shape[0]
            if max_out is None or out > max_out:
                max_out = out
    return max_out


def load_split_json(split_path: str):
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    # expected keys: test_paths, (optional) test_labels, (optional) label_encoder_classes
    test_paths = split.get("test_paths", [])
    test_labels = split.get("test_labels", None)
    classes = split.get("label_encoder_classes", None)
    return test_paths, test_labels, classes


def coerce_int_list(x: Sequence):
    try:
        return [int(v) for v in x]
    except Exception:
        return None


def main(args):
    set_deterministic(43)

    # --- device selection: prefer MPS (Apple GPU) -> CUDA -> CPU ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Optional: allow unsupported ops to fall back to CPU instead of crashing
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 1) Load checkpoint first to know classes/num_classes
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("model_state", ckpt)

    ckpt_classes = None
    if "label_encoder_classes" in ckpt:
        ckpt_classes = [str(c).upper() for c in ckpt["label_encoder_classes"]]
        print(f"[Info] classes from checkpoint: {ckpt_classes}")
    else:
        inferred = infer_num_classes_from_state(state)
        print(f"[Info] inferred num_classes from checkpoint weights: {inferred}")

    # 2) Decide test set source
    used_paths, used_label_names, classes_for_le, test_labels_int = None, None, None, None

    if args.split_json:
        print(f"[Info] Loading fixed split from: {args.split_json}")
        split_paths, split_labels, split_classes = load_split_json(args.split_json)

        # sanitize paths (filter missing files)
        missing = [p for p in split_paths if not os.path.exists(p)]
        if missing:
            print(f"[Warn] {len(missing)} files in split are missing on disk. They will be skipped.")
        split_paths = [p for p in split_paths if os.path.exists(p)]

        # classes priority: split_json > checkpoint > classes_json > tokens order
        if split_classes:
            classes_for_le = [str(c).upper() for c in split_classes]
        elif ckpt_classes is not None:
            classes_for_le = ckpt_classes
        elif args.classes_json and os.path.exists(args.classes_json):
            with open(args.classes_json, "r", encoding="utf-8") as f:
                classes_for_le = [str(c).upper() for c in json.load(f)]

        # if labels are provided as ints: use them directly
        test_labels_int = coerce_int_list(split_labels) if split_labels is not None else None
        if test_labels_int is not None:
            if classes_for_le is None:
                raise RuntimeError("split_json provides integer labels but no class list; "
                                   "please include 'label_encoder_classes' in the split or pass --classes_json.")
            used_paths = split_paths
            # names will be derived after we build LabelEncoder
        else:
            # labels not given (or given as strings but we don't rely on it) → derive from filename like before
            used_paths = split_paths

    else:
        # 2b) Build from test_dir and parse labels from filename
        test_paths = list_videos(args.test_dir)
        if not test_paths:
            raise RuntimeError(f"No videos found under {args.test_dir}")
        default_tokens = ["ASD", "DHS", "LCS", "HIPOA"]
        tokens = [t.strip() for t in args.tokens.split(",")] if args.tokens else default_tokens
        regex = args.regex if args.regex else r"(ASD|DHS|LCS|HIPOA)"

        label_names = []
        for p in test_paths:
            lab = extract_label_from_filename(os.path.basename(p), regex, tokens)
            label_names.append(lab)

        filtered = [(p, lab) for p, lab in zip(test_paths, label_names) if lab is not None]
        if not filtered:
            raise RuntimeError("No test samples matched label pattern. Adjust --regex/--tokens.")
        used_paths, used_label_names = map(list, zip(*filtered))

        # classes priority: checkpoint > classes_json > tokens order
        if ckpt_classes is not None:
            classes_for_le = ckpt_classes
        elif args.classes_json and os.path.exists(args.classes_json):
            with open(args.classes_json, "r", encoding="utf-8") as f:
                classes_for_le = [str(c).upper() for c in json.load(f)]
        else:
            classes_for_le = [t.strip().upper() for t in (args.tokens.split(",") if args.tokens else ["ASD","DHS","LCS","HIPOA"])]

    # 3) Build LabelEncoder
    le = LabelEncoder()
    if classes_for_le is not None:
        le.fit(classes_for_le)
    else:
        # ultimate fallback: use unique labels from used_label_names
        if used_label_names is None:
            raise RuntimeError("Cannot determine classes for LabelEncoder. Provide --classes_json or store classes in split/checkpoint.")
        le.fit(sorted(list({n.upper() for n in used_label_names})))

    # 4) Build test label indices
    if test_labels_int is not None:
        # use given integer labels (assumed to match 'classes_for_le' order)
        test_labels = np.array(test_labels_int, dtype=int)
        print(f"[Info] Loaded {len(test_labels)} integer labels from split.")
    else:
        if used_label_names is None:
            # derive from filenames using same logic as above
            default_tokens = ["ASD", "DHS", "LCS", "HIPOA"]
            tokens = [t.strip() for t in args.tokens.split(",")] if args.tokens else default_tokens
            regex = args.regex if args.regex else r"(ASD|DHS|LCS|HIPOA)"
            used_label_names = []
            for p in used_paths:
                lab = extract_label_from_filename(os.path.basename(p), regex, tokens)
                used_label_names.append(lab)
        # keep only samples whose label exists in classes
        tmp_paths, tmp_names = [], []
        for p, name in zip(used_paths, used_label_names):
            if name is not None and name.upper() in le.classes_:
                tmp_paths.append(p)
                tmp_names.append(name.upper())
        used_paths, used_label_names = tmp_paths, tmp_names
        if not used_paths:
            raise RuntimeError("After aligning labels with classes, zero usable samples remained.")
        test_labels = le.transform(used_label_names)

    num_classes = len(le.classes_)
    print(f"LabelEncoder fitted with {num_classes} classes: {list(le.classes_)}")
    if args.split_json:
        print(f"Using fixed test split from: {args.split_json}")
    else:
        print(f"Loading independent test set from: {args.test_dir}")
    print(f"Creating test dataset with {len(used_paths)} videos.")

    # 5) Dataset / Loader
    test_dataset = VideoDataset(used_paths, test_labels, num_frames=args.num_frames)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # 6) Model & load weights
    print("Building 3D-CNN model architecture...")
    model = Simple3DCNN(num_classes=num_classes).to(device).to(torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[Warn] missing:", missing, "unexpected:", unexpected)

    # 7) Evaluate
    accuracy, report_df, cm_fig = evaluate_like_before(model, test_loader, device, le)

    print("\n" + "="*50)
    print("      Final 3D-CNN Model Evaluation on Test Set")
    print("="*50)
    print(f"\nModel Weights: {args.weights}")
    if args.split_json:
        print(f"Split JSON:   {args.split_json}")
    else:
        print(f"Test Set Path:{args.test_dir}")
    print(f"\nAccuracy on Test Set: {accuracy:.2%}")
    print("\nDetailed Classification Report:\n")
    print(report_df)
    print("="*50)

    # 8) Save report & confusion matrix
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(os.path.normpath(args.test_dir if args.test_dir else "test"))
    if args.split_json:
        base = os.path.splitext(os.path.basename(args.split_json))[0]
    report_save_path = os.path.join(output_dir, f"3dcnn_report_{base}.csv")
    cm_save_path = os.path.join(output_dir, f"3dcnn_cm_{base}.png")
    report_df.to_csv(report_save_path)
    cm_fig.savefig(cm_save_path)
    print(f"Classification report saved to: {report_save_path}")
    print(f"Confusion matrix saved to: {cm_save_path}")

    # 9) Model summary (optional)
    try:
        from torchinfo import summary
        dummy_input = torch.randn(1, 3, args.num_frames, 112, 112, dtype=torch.float32, device=device)
        print("\nModel Summary:")
        summary(model, input_data=dummy_input)
    except Exception as e:
        print(f"[Note] torchinfo.summary not available or failed: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Deterministic evaluation with MPS-first. Supports reading fixed split JSON or filename-parsed labels."
    )
    ap.add_argument("--weights", required=True, help="Path to best_model.pth")
    ap.add_argument("--test_dir", default="", help="Folder containing test videos (*.mp4, etc.) (ignored if --split_json is given)")
    ap.add_argument("--split_json", default="", help="JSON produced during training containing test_paths/test_labels/label_encoder_classes")
    ap.add_argument("--classes_json", default="", help="Optional JSON list of class names in TRAINING order (used if needed).")
    ap.add_argument("--regex", default="", help="Regex to extract label from filename (default: ASD|DHS|LCS|HipOA).")
    ap.add_argument("--tokens", default="", help="Comma-separated tokens as fallback search (default: ASD,DHS,LCS,HipOA).")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    main(args)
