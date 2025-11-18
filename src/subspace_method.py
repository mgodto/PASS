# src/subspace_method.py

import numpy as np
import os
import subprocess
from ezc3d import c3d

def convert_npy_to_c3d(npy_path: str, c3d_path: str):
    data = np.load(npy_path)
    c3d_data = data.transpose((2, 1, 0))
    
    corrected_pose_data = np.copy(c3d_data)
    original_x = c3d_data[0, :, :]
    original_y = c3d_data[1, :, :]
    original_z = c3d_data[2, :, :]
    
    corrected_pose_data[0, :, :] = original_x * -1
    corrected_pose_data[1, :, :] = original_z
    corrected_pose_data[2, :, :] = original_y * -1
    
    c = c3d()
    c['data']['points'] = corrected_pose_data
    
    num_landmarks = corrected_pose_data.shape[1]
    point_labels = [f'JOINT_{i+1}' for i in range(num_landmarks)]
    c['parameters']['POINT']['LABELS']['value'] = point_labels
    c['parameters']['POINT']['RATE']['value'] = [120.0]

    c.write(c3d_path)
    print(f"Successfully converted and saved to {c3d_path}")

def run_visual_contribution_scripts(c3d_path: str, script_dir: str, gif_dir: str, npy_dir: str):
    scripts_to_run = {
        "visual_contribution_first.py": "_first",
        "visual_contribution_geodesic.py": "_geodesic",
        "visual_contribution_second.py": "_second"
    }
    
    abs_c3d_path = os.path.abspath(c3d_path)
    abs_gif_dir = os.path.abspath(gif_dir)
    abs_npy_dir = os.path.abspath(npy_dir)

    input_basename = os.path.splitext(os.path.basename(c3d_path))[0]

    for script_name, suffix in scripts_to_run.items():
        script_path = os.path.join(script_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"Warning: Script not found, skipping: {script_name}")
            continue
            
        # --- ここからが追加部分 ---
        
        # このスクリプトが生成するはずの出力ファイルパスを事前に構築
        gif_filename = f"{input_basename}{suffix}.gif"
        gif_save_path = os.path.join(abs_gif_dir, gif_filename)
        
        npy_filename = f"{input_basename}{suffix}_mag.npy"
        npy_save_path = os.path.join(abs_npy_dir, npy_filename)
        
        # 対応するGIFとNPYの両方が既に存在するかチェック
        if os.path.exists(gif_save_path) and os.path.exists(npy_save_path):
            print(f"Skipping script {script_name}, output files already exist.")
            continue  # このスクリプトの実行をスキップして、ループの次へ進む
            
        # --- 追加はここまで ---
            
        print(f"--- Running script: {script_name} for {os.path.basename(c3d_path)} ---")
        
        command = [
            "python", 
            script_path, 
            "--input", abs_c3d_path, 
            "--output_gif", gif_save_path,
            "--output_npy", npy_save_path
        ]
        
        print(f"Executing command: {' '.join(command)}")
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_name}: {e}")