# src/config.py

# --- Path Settings ---
RAW_VIDEO_DIR = 'data/segmentation_dataset_512/alldata/train'
PROCESSED_SKELETON_DIR = 'data/processed_skeletons'
C3D_DIR = 'data/c3d_files'
GIF_DIR = 'results/gifs'
SUBSPACE_NPY_DIR = 'results/subspace_npy'
PARTITION_NPY_DIR = "/Users/gaoji/projects/human_gait/results/partition_npy"

# 外部腳本所在的目錄
VISUAL_CONTRIBUTION_SCRIPT_DIR = 'src/cv_motion3d_public-main/src'

# --- Final Dataset Paths ---
TRAIN_PATH = 'results/train'
SVM_FEATURES_PATH = 'results/train/svm_features.npy'
LABELS_PATH = 'results/train/labels.npy'
STGCN_PATHS_PATH = 'results/train/stgcn_paths.npy'
