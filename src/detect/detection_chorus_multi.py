# detect_chorus_multi.py

import librosa
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from paths import TEST_AUDIO_DIR, MODELS_DIR
import joblib 
import json   
import matplotlib.pyplot as plt
import librosa.display


plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False 

# 設定ファイルとモデルファイルのパス
MODEL_PATH = os.path.join(MODELS_DIR, 'trained_random_forest_chorus_detector.pkl')
CONFIG_PATH = os.path.join(MODELS_DIR, 'feature_extraction_config.json')

# サビ検出したい複数の楽曲のパスリスト
AUDIO_DIR = TEST_AUDIO_DIR
NUM_SONGS_TO_DETECT = 3 
SONG_PATHS_TO_DETECT = [os.path.join(AUDIO_DIR, f"song_{i:03d}.wav") for i in range(1, NUM_SONGS_TO_DETECT + 1)]


# 学習済みモデルの読み込み
try:
    model = joblib.load(MODEL_PATH)
    print(f"  モデルを '{MODEL_PATH}' からロード")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please run 'train_and_save_model.py' first.")
    exit()

try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    SR = config['SR']
    N_MFCC = config['N_MFCC']
    HOP_LENGTH = config['HOP_LENGTH']
    print(f"  特徴量抽出設定を '{CONFIG_PATH}' からロード: SR={SR}, N_MFCC={N_MFCC}, HOP_LENGTH={HOP_LENGTH}")
except FileNotFoundError:
    print(f"Error: Config file not found at {CONFIG_PATH}. Please run 'train_and_save_model.py' first.")
    exit()

#サビ検出ループ

for song_path in SONG_PATHS_TO_DETECT:
    print(f"\n楽曲: {os.path.basename(song_path)} ")

    print(f"\n'{os.path.basename(song_path)}' の特徴量抽出開始")
    try:
        y_new, sr_new = librosa.load(song_path, sr=SR)
        print(f"  Audio loaded. Sample rate: {sr_new} Hz, Duration: {librosa.get_duration(y=y_new, sr=sr_new):.2f} seconds")
    except FileNotFoundError:
        print(f"  Error: Audio file not found at {song_path}. Please check the path.")
        continue 
    except Exception as e:
        print(f"  Error loading audio file {song_path}: {e}. Skipping.")
        continue 

    mfccs_new = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=2048)
    X_new = mfccs_new.T
    print(f"  MFCCs extracted for new song. Shape: {X_new.shape}")

    # モデルによる予測
    print("\n予測開始")
    predictions = model.predict(X_new)
    print(f"  予測結果のフレーム数: {len(predictions)}")
    print(f"  予測されたサビのフレーム数: {np.sum(predictions == 1)}")

    #サビ区間の特定

    frame_times = librosa.frames_to_time(np.arange(len(predictions)), sr=sr_new, hop_length=HOP_LENGTH)

    detected_choruses = []
    current_chorus_start_frame = -1
    # 最小サビ継続時間と最小非サビ間隔
    min_chorus_duration_sec = 10.0
    min_non_chorus_gap_sec = 1.0

    for i, pred in enumerate(predictions):
        if pred == 1:
            if current_chorus_start_frame == -1:
                current_chorus_start_frame = i
        else:
            if current_chorus_start_frame != -1:
                chorus_end_frame = i
                detected_choruses.append((current_chorus_start_frame, chorus_end_frame))
                current_chorus_start_frame = -1

    if current_chorus_start_frame != -1:
        detected_choruses.append((current_chorus_start_frame, len(predictions)))

    #短いサビ区間の結合
    merged_choruses = []
    if detected_choruses:
        current_start_frame, current_end_frame = detected_choruses[0]
        for i in range(1, len(detected_choruses)):
            next_start_frame, next_end_frame = detected_choruses[i]
            gap_duration_frames = next_start_frame - current_end_frame
            gap_duration_sec = librosa.frames_to_time(gap_duration_frames, sr=sr_new, hop_length=HOP_LENGTH)

            if gap_duration_sec < min_non_chorus_gap_sec:
                current_end_frame = next_end_frame
            else:
                merged_choruses.append((current_start_frame, current_end_frame))
                current_start_frame = next_start_frame
                current_end_frame = next_end_frame
        merged_choruses.append((current_start_frame, current_end_frame))
    
    #最小継続時間でのフィルタリング
    choruses_with_duration = []
    for start_f, end_f in merged_choruses:
        start_t = librosa.frames_to_time(start_f, sr=sr_new, hop_length=HOP_LENGTH)
        end_t = librosa.frames_to_time(end_f, sr=sr_new, hop_length=HOP_LENGTH)
        duration_sec = end_t - start_t
        
        if duration_sec >= min_chorus_duration_sec:
            choruses_with_duration.append((start_t, end_t, duration_sec))

    # 継続時間の長い順にソート
    choruses_with_duration.sort(key=lambda x: x[2], reverse=True)
    final_filtered_choruses = choruses_with_duration[:4]
    final_filtered_choruses.sort(key=lambda x: x[0])


    print("\n検出されたサビ区間")
    if final_filtered_choruses:
        for i, (start_t, end_t, duration_t) in enumerate(final_filtered_choruses):
            print(f"  サビ {i+1}: {start_t:.2f}秒 - {end_t:.2f}秒 ({duration_t:.2f}秒)")
    else:
        print("  サビは検出されませんでした。")

    # --- 結果の可視化 (オプション) ---
    # 各曲ごとにグラフを表示するかどうかは、必要に応じてコメントアウトしてください。
    # 大量の曲を処理する場合、毎回グラフが表示されると不便かもしれません。
    print("\n結果の可視化")
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y_new, sr=sr_new, alpha=0.6)
    plt.title(f'Audio Waveform: {os.path.basename(song_path)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(frame_times, predictions, label='Predicted Chorus (1=Chorus, 0=Non-Chorus)', color='gray', alpha=0.7)
    for start_t, end_t, _ in final_filtered_choruses:
        plt.axvspan(start_t, end_t, color='red', alpha=0.3, label='Detected Chorus' if start_t == final_filtered_choruses[0][0] else "")
    plt.title('Predicted Chorus Segments (Filtered)')
    plt.xlabel('Time (s)')
    plt.ylabel('Label')
    plt.yticks([0, 1], ['Non-Chorus', 'Chorus'])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
