# main.py
import os
import glob
import pandas as pd
import numpy as np
import argparse
from colorama import Fore, Style, init # <-- 這行を追加
from src import config as cfg
from src import data_processing, subspace_method, visualize, train

def run_preprocessing(args):
    print(f"\n{Style.BRIGHT}{Fore.YELLOW}[{i}/{total_files}] Processing video: {basename}{Style.RESET_ALL}")
    os.makedirs(cfg.PROCESSED_SKELETON_DIR, exist_ok=True)
    video_paths = glob.glob(os.path.join(cfg.RAW_VIDEO_DIR, '*'))
    
    if not video_paths:
        print(f"Error: No video files found in the directory: {os.path.abspath(cfg.RAW_VIDEO_DIR)}")
        return

    print(f"Found {len(video_paths)} videos to process.")
    
    for video_path in video_paths:
        basename = os.path.basename(video_path)
        output_filename = os.path.splitext(basename)[0] + '.npy'
        output_path = os.path.join(cfg.PROCESSED_SKELETON_DIR, output_filename)

        if not args.force and os.path.exists(output_path):
            print(f"Skipping {basename}, already processed.")
            continue
        
        print(f"--- STARTING PROCESSING FOR: {basename} ---")
        try:
            cleaned_skeleton = data_processing.process_video_to_skeleton(video_path)
            if cleaned_skeleton is not None:
                print(f"Saving data with shape {cleaned_skeleton.shape} to {output_path}")
                np.save(output_path, cleaned_skeleton)
            else:
                print(f"Warning: Processing failed for {basename}. No data to save.")
        except Exception as e:
            print(f"--- FAILED TO PROCESS {basename} due to an error: {e} ---")
            continue
            
    print("--- Preprocessing complete. ---")

def run_subspace_analysis(args):
    print("\n--- Running Step 2: Subspace Method Analysis ---")
    os.makedirs(cfg.C3D_DIR, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)
    os.makedirs(args.npy_dir, exist_ok=True)
    
    if args.file:
        single_file_path = os.path.join(cfg.PROCESSED_SKELETON_DIR, args.file)
        if not os.path.exists(single_file_path):
            print(f"Error: Specified file not found: {single_file_path}")
            return
        npy_files = [single_file_path]
        print(f"--- Running in single-file mode for: {args.file} ---")
    else:
        npy_files = glob.glob(os.path.join(cfg.PROCESSED_SKELETON_DIR, '*.npy'))
        print(f"--- Running in batch mode for all {len(npy_files)} files ---")

    if not npy_files:
        print("Error: No processed .npy files found to process.")
        return

    total_files = len(npy_files)
    for i, npy_path in enumerate(npy_files, 1):
        basename = os.path.basename(npy_path)
        
        # --- 修正2：加入進度提示 ---
        print(f"\n{Style.BRIGHT}{Fore.CYAN}[{i}/{total_files}] Analyzing file: {basename}{Style.RESET_ALL}")
        
        filename_no_ext = os.path.splitext(basename)[0]
        
        check_gif_path = os.path.join(args.gif_dir, f"{filename_no_ext}_first.gif")
        if not args.force and os.path.exists(check_gif_path):
            print(f"--> Skipping, output files already exist.")
            continue
        
        c3d_path = os.path.join(cfg.C3D_DIR, filename_no_ext + '.c3d')
        
        subspace_method.convert_npy_to_c3d(npy_path, c3d_path)
        
        subspace_method.run_visual_contribution_scripts(
            c3d_path=c3d_path, 
            script_dir=cfg.VISUAL_CONTRIBUTION_SCRIPT_DIR, 
            gif_dir=args.gif_dir,
            npy_dir=args.npy_dir
        )
    print("\n--- Subspace method analysis complete. ---")

def run_visualization(args):
    print(f"--- Running Step 4: Visualizing Skeleton File ---")
    filepath = os.path.join(cfg.PROCESSED_SKELETON_DIR, args.filename)
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return
    skeleton_data = np.load(filepath)
    visualize.create_skeleton_animation(skeleton_data)


def run_training(args):
    """步驟3: 訓練分類模型"""
    print("\n--- Running Step 3: Model Training ---")
    # 直接呼叫 train 模組中的函數，不再需要傳遞參數
    train.train_and_evaluate()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gait Analysis Project Pipeline.")
    subparsers = parser.add_subparsers(dest='action', required=True, help="Available actions")

    parser_preprocess = subparsers.add_parser('preprocess', help='Run data preprocessing pipeline.')
    parser_preprocess.add_argument('--force', action='store_true', help='Force reprocessing of all videos.')
    parser_preprocess.set_defaults(func=run_preprocessing)


    parser_subspace = subparsers.add_parser('subspace', help='Run subspace method analysis.')
    parser_subspace.add_argument('--file', type=str, default=None, help='Specify a single .npy file name to process.')
    parser_subspace.add_argument('--gif_dir', type=str, default=cfg.GIF_DIR, help=f'Directory to save output GIFs (default: {cfg.GIF_DIR})')
    parser_subspace.add_argument('--npy_dir', type=str, default=cfg.SUBSPACE_NPY_DIR, help=f'Directory to save subspace npy files (default: {cfg.SUBSPACE_NPY_DIR})')
    parser_subspace.add_argument('--force', action='store_true', help='Force reprocessing, even if output files exist.')
    parser_subspace.set_defaults(func=run_subspace_analysis)
    
    parser_visualize = subparsers.add_parser('visualize', help='Visualize a processed skeleton .npy file.')
    parser_visualize.add_argument('filename', type=str, help='Name of the .npy file to visualize.')
    parser_visualize.set_defaults(func=run_visualization)
    
    # 'train' 
    parser_train = subparsers.add_parser('train', help='Train a classification model using paths from config.py.')
    parser_train.set_defaults(func=run_training)

    args = parser.parse_args()
    args.func(args)