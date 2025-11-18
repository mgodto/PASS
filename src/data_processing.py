import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# MediaPipe Poseオブジェクトの初期化
mp_pose = mp.solutions.pose

def interpolate_data(data_array: np.ndarray) -> np.ndarray:
    """3D姿勢データ内のNaN値を線形補間する"""
    num_frames, num_landmarks, num_coords = data_array.shape
    for landmark_idx in range(num_landmarks):
        for coord_idx in range(num_coords):
            time_series = data_array[:, landmark_idx, coord_idx]
            s = pd.Series(time_series)
            s_interpolated = s.interpolate(method='linear', limit_direction='forward')
            s_filled = s_interpolated.bfill().ffill()
            data_array[:, landmark_idx, coord_idx] = s_filled.values
    return data_array

def smooth_data(data_array: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golayフィルターでデータを平滑化する"""
    if window_length % 2 == 0:
        window_length += 1
    
    smoothed_data = np.copy(data_array)
    num_frames, num_landmarks, num_coords = data_array.shape

    for landmark_idx in range(num_landmarks):
        for coord_idx in range(num_coords):
            time_series = data_array[:, landmark_idx, coord_idx]
            smoothed_data[:, landmark_idx, coord_idx] = savgol_filter(time_series, window_length, polyorder)
    return smoothed_data


def process_video_to_skeleton(video_path: str) -> np.ndarray:
    """
    単一の動画ファイルを読み込み、MediaPipeで骨格抽出し、
    データクレンジングパイプラインを実行して、最終的なNumpy配列を返す。
    """
    all_landmarks_list = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return None

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks.landmark
                current_frame_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                all_landmarks_list.append(current_frame_data)
            else:
                all_landmarks_list.append(None)
        cap.release()

    if not all_landmarks_list:
        print(f"No frames processed for video: {video_path}")
        return None

    # リストをNumpy配列に変換
    num_landmarks = 33 # MediaPipe Poseのデフォルト
    length = len(all_landmarks_list)
    data_with_gaps = np.full((length, num_landmarks, 3), np.nan)
    for i, frame_data in enumerate(all_landmarks_list):
        if frame_data is not None:
            data_with_gaps[i] = frame_data

    # corrected_data = correct_coordinates(data_with_gaps)

    # データクレンジングパイプライン
    interpolated_data = interpolate_data(data_with_gaps)
    smoothed_data = smooth_data(interpolated_data)
    
    return smoothed_data

# このファイルが直接実行されたときにテストするためのコード
if __name__ == '__main__':
    # ここにテスト用の動画ファイルのパスを書く
    test_video_path = 'path/to/your/test_video.mp4' 
    print(f"Testing data processing for: {test_video_path}")
    
    cleaned_skeleton = process_video_to_skeleton(test_video_path)
    
    if cleaned_skeleton is not None:
        print(f"Processing successful. Output shape: {cleaned_skeleton.shape}")
        print(f"Check for NaNs: {np.isnan(cleaned_skeleton).any()}")