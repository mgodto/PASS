
import os
import argparse
import json
import glob
import re
import torch
import numpy as np
import random
from typing import List

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

try:
    from video_dataset_strict import VideoDatasetStrict as VideoDataset
except ImportError:
    from src.video_dataset import VideoDataset

from src.cnn_models import Simple3DCNN
from src.cnn_engine import evaluate

VALID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
def is_video(p): return os.path.splitext(p)[1].lower() in VALID_EXTS

def list_videos(folder):
    return sorted([p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if is_video(p)])

def set_deterministic(seed=43):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def infer_num_classes_from_state(state_dict):
    for key in ["classifier.4.weight", "classifier.3.weight", "fc.weight", "head.weight"]:
        if key in state_dict:
            return state_dict[key].shape[0]
    max_out = None
    for k, w in state_dict.items():
        if k.endswith(".weight") and hasattr(w, "ndim") and w.ndim == 2:
            out = w.shape[0]
            if max_out is None or out > max_out:
                max_out = out
    return max_out

def extract_label_from_filename(fname: str, pattern: str, alt_tokens: List[str]):
    # pattern example: r"(ASD|DHS|LCS|HipOA)"
    m = re.search(pattern, fname, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper() if m.groups() else m.group(0).upper()
    # fallback: search any alt token
    upper = fname.upper()
    for t in alt_tokens:
        if t.upper() in upper:
            return t.upper()
    return None

def main(args):
    set_deterministic(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt.get("model_state", ckpt)
    if "label_encoder_classes" in ckpt:
        ckpt_classes = [str(c) for c in list(ckpt["label_encoder_classes"])]
        print(f"[Info] classes from checkpoint: {ckpt_classes}")
    else:
        ckpt_classes = None
        n_from_ckpt = infer_num_classes_from_state(state)
        print(f"[Info] inferred num_classes from checkpoint weights: {n_from_ckpt}")

    # 2) Build deterministic test file list
    test_paths = list_videos(args.test_dir)
    if len(test_paths) == 0:
        raise RuntimeError(f"No videos found under {args.test_dir}")

    # 3) Build label names from filenames (your case)
    token_list = [t.strip() for t in args.tokens.split(",")] if args.tokens else ["ASD","DHS","LCS","HIPOA"]
    regex = args.regex if args.regex else r"(ASD|DHS|LCS|HIPOA)"
    file_labels = []
    for p in test_paths:
        name = os.path.basename(p)
        lab = extract_label_from_filename(name, regex, token_list)
        file_labels.append(lab)

    # filter samples with unknown labels
    filtered_paths, filtered_label_names = [], []
    for p, lab in zip(test_paths, file_labels):
        if lab is None:
            continue
        filtered_paths.append(p)
        filtered_label_names.append(lab)

    if len(filtered_paths) == 0:
        raise RuntimeError("No test samples matched your label pattern. "
                           "Please adjust --regex or --tokens to match filenames.")

    # 4) Build LabelEncoder locked to checkpoint classes order (preferred)
    if ckpt_classes is not None:
        classes = [str(c).upper() for c in ckpt_classes]
    elif args.classes_json and os.path.exists(args.classes_json):
        with open(args.classes_json, "r", encoding="utf-8") as f:
            classes = [str(c).upper() for c in json.load(f)]
    else:
        # fallback: use tokens order (uppercased)
        classes = [t.upper() for t in token_list]
        if len(set(classes)) == 1 and classes[0] == "":
            # ultimate fallback: unique labels from data, sorted
            classes = sorted(list({lab for lab in filtered_label_names if lab is not None}))

    le = LabelEncoder(); le.fit(classes)

    # keep only samples whose label exists in classes
    used_paths, used_labels = [], []
    skipped = 0
    for p, lab in zip(filtered_paths, filtered_label_names):
        if lab in le.classes_:
            used_paths.append(p)
            used_labels.append(lab)
        else:
            skipped += 1
    if skipped:
        print(f"[Warn] {skipped} samples skipped because their label not in encoder classes: {set(filtered_label_names) - set(le.classes_)}")

    if len(used_paths) == 0:
        raise RuntimeError("After aligning labels with classes, zero samples remain. "
                           "Double-check your checkpoint class names vs filename tokens. "
                           "Consider passing --classes_json with the exact training order.")

    test_labels = le.transform(used_labels)
    num_classes = len(le.classes_)
    print(f"[Info] test samples used={len(used_paths)}, classes={list(le.classes_)}")

    # 5) DataLoader (no shuffle)
    test_ds = VideoDataset(used_paths, test_labels, num_frames=args.num_frames)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 6) Build model with SAME num_classes as encoder and load weights
    model = Simple3DCNN(num_classes=num_classes).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[Warn] missing:", missing, "unexpected:", unexpected)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # 7) Evaluate once (deterministic)
    avg_loss, acc, report, _ = evaluate(model, test_loader, loss_fn, device, le)
    print(f"[Result] loss={avg_loss:.4f}, acc={acc*100:.2f}%")
    print(report)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Deterministic evaluation of a trained 3D-CNN (labels from filenames).")
    ap.add_argument("--weights", required=True, help="Path to best_model.pth")
    ap.add_argument("--test_dir", required=True, help="Folder containing test videos (*.mp4, etc.)")
    ap.add_argument("--classes_json", default="", help="Optional JSON file of class names in TRAINING order.")
    ap.add_argument("--regex", default="", help="Regex to extract label from filename. Default matches ASD/DHS/LCS/HipOA (case-insensitive).")
    ap.add_argument("--tokens", default="", help="Comma-separated fallback tokens to search in filename (e.g., ASD,DHS,LCS,HipOA).")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    main(args)
