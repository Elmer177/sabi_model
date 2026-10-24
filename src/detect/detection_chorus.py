import librosa
import numpy as np
import os
import joblib # モデルの読み込みに必要
import json   # 設定の読み込みに必要
import matplotlib.pyplot as plt
import librosa.display

#設定ファイルとモデルファイルのパス
MODEL_PATH = 'trained_random_forest_chorus_detector.pkl'
CONFIG_PATH = 'feature_extraction_config.json'

#サビ検出したい曲のパス
NEW_SONG_PATH = 'music_dataset/audio/new_song_example.wav' 

#学習済みモデルと設定の読み込み
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

#新しい楽曲からの特徴量抽出
print(f"\n'{os.path.basename(NEW_SONG_PATH)}' の特徴量抽出開始")
try:
    y_new, sr_new = librosa.load(NEW_SONG_PATH, sr=SR)
    print(f"  Audio loaded. Sample rate: {sr_new} Hz, Duration: {librosa.get_duration(y=y_new, sr=sr_new):.2f} seconds")
except FileNotFoundError:
    print(f"  Error: New song file not found at {NEW_SONG_PATH}. Please check the path.")
    exit()
except Exception as e:
    print(f"  Error loading new song file {NEW_SONG_PATH}: {e}.")
    exit()

mfccs_new = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=2048)
X_new = mfccs_new.T
print(f"  MFCCs extracted for new song. Shape: {X_new.shape}")

#モデルによる予測
print("\n予測開始")
predictions = model.predict(X_new)
print(f"  予測結果のフレーム数: {len(predictions)}")
print(f"  予測されたサビのフレーム数: {np.sum(predictions == 1)}")

#サビ区間の特定
print("\nサビ区間の特定")

frame_times = librosa.frames_to_time(np.arange(len(predictions)), sr=sr_new, hop_length=HOP_LENGTH)

detected_choruses = []
current_chorus_start_frame = -1
min_chorus_duration_sec = 10.0 # 最小サビ継続時間（秒）
min_non_chorus_gap_sec = 1.0   # 最小非サビ間隔（秒）

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

final_choruses = []
if detected_choruses:
    current_start_frame, current_end_frame = detected_choruses[0]
    for i in range(1, len(detected_choruses)):
        next_start_frame, next_end_frame = detected_choruses[i]
        gap_duration_frames = next_start_frame - current_end_frame
        gap_duration_sec = librosa.frames_to_time(gap_duration_frames, sr=sr_new, hop_length=HOP_LENGTH)
        if gap_duration_sec < min_non_chorus_gap_sec:
            current_end_frame = next_end_frame
        else:
            final_choruses.append((current_start_frame, current_end_frame))
            current_start_frame = next_start_frame
            current_end_frame = next_end_frame
    final_choruses.append((current_start_frame, current_end_frame))

filtered_choruses = []
for start_f, end_f in final_choruses:
    duration_sec = librosa.frames_to_time(end_f - start_f, sr=sr_new, hop_length=HOP_LENGTH)
    if duration_sec >= min_chorus_duration_sec:
        filtered_choruses.append((librosa.frames_to_time(start_f, sr=sr_new, hop_length=HOP_LENGTH),
                                  librosa.frames_to_time(end_f, sr=sr_new, hop_length=HOP_LENGTH)))

print("\n検出されたサビ区間")
if filtered_choruses:
    for i, (start_t, end_t) in enumerate(filtered_choruses):
        print(f"  サビ {i+1}: {start_t:.2f}秒 - {end_t:.2f}秒 ({end_t - start_t:.2f}秒)")
else:
    print("サビは検出されませんでした")

# --- 結果の可視化 (オプション) ---
print("\n結果の可視化")
plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y_new, sr=sr_new, alpha=0.6)
plt.title(f'Audio Waveform: {os.path.basename(NEW_SONG_PATH)}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(frame_times, predictions, label='Predicted Chorus (1=Chorus, 0=Non-Chorus)', color='gray', alpha=0.7)
for start_t, end_t in filtered_choruses:
    plt.axvspan(start_t, end_t, color='red', alpha=0.3, label='Detected Chorus' if start_t == filtered_choruses[0][0] else "")
plt.title('Predicted Chorus Segments')
plt.xlabel('Time (s)')
plt.ylabel('Label')
plt.yticks([0, 1], ['Non-Chorus', 'Chorus'])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
