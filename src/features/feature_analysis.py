import librosa
import numpy as np
import pandas as pd
import os

#変数の宣言
DATASET_ROOT = 'music_dataset' # データセットのルートディレクトリ
AUDIO_DIR = os.path.join(DATASET_ROOT, 'audio')
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, 'annotation')
NUM_SONGS = 20     #曲数
SR = 22050        # サンプリングレート
N_MFCC = 13       # MFCCの次元数
HOP_LENGTH = 512  # サンプリングレート

# 特徴量とラベルを格納するリスト
all_features = [] # 全楽曲の特徴量
all_labels = []   # 全楽曲のラベル
song_names = []   # 処理した楽曲名

print("特徴量抽出を開始します")

#ファイル読み込み
for i in range(1, NUM_SONGS + 1):
    song_name = f"song_{i:03d}" 
    audio_path = os.path.join(AUDIO_DIR, f"{song_name}.wav")
    annotation_path = os.path.join(ANNOTATIONS_DIR, f"label_{i:03d}.txt")

    print(f"\nProcessing: {song_name}")

    #エラー表示
    try:
        y, sr_loaded = librosa.load(audio_path, sr=SR)
        print(f"  Audio loaded. Sample rate: {sr_loaded} Hz, Duration: {librosa.get_duration(y=y, sr=sr_loaded):.2f} seconds")
    except FileNotFoundError:
        print(f"  Error: Audio file not found at {audio_path}. Skipping.")
        continue
    except Exception as e:
        print(f"  Error loading audio file {audio_path}: {e}. Skipping.")
        continue

    # MFCC特徴量の抽出
 
    mfccs = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=2048)
    # mfccsの形状は (n_mfcc, フレーム数)
    print(f"  MFCCs extracted. Shape: {mfccs.shape}")

    #アノテーションファイルの読み込みとラベルの生成
    try:
        annotations_df = pd.read_csv(annotation_path, sep='\t', header=None, names=['start_time', 'end_time', 'label'])
        print(f"  Annotations loaded. {len(annotations_df)} entries found.")
    except FileNotFoundError:
        print(f"  Error: Annotation file not found at {annotation_path}. Cannot generate labels. Skipping.")
        continue
    except Exception as e:
        print(f"  Error loading annotation file {annotation_path}: {e}. Skipping.")
        continue

    # 各MFCCフレームに対応するラベル (サビ: 1, 非サビ: 0) を生成
    frame_times = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=sr_loaded, hop_length=HOP_LENGTH)
    labels = np.zeros(mfccs.shape[1], dtype=int) # 全て非サビ(0)で初期化

    sabi_found = False
    for _, ann_row in annotations_df.iterrows():
        if ann_row['label'].strip().lower() == 'sabi': # ラベル名が 'sabi' の場合を抽出
            sabi_found = True
            # サビ区間の開始・終了時刻に対応するフレームインデックスを計算
            start_frame = librosa.time_to_frames(ann_row['start_time'], sr=sr_loaded, hop_length=HOP_LENGTH)
            end_frame = librosa.time_to_frames(ann_row['end_time'], sr=sr_loaded, hop_length=HOP_LENGTH)
            
            # 範囲を適切に調整 (mfccsのフレーム数を超えないように)
            start_frame = max(0, start_frame)
            end_frame = min(mfccs.shape[1], end_frame)

            # サビ区間のフレームを1に設定
            labels[start_frame:end_frame] = 1

    if not sabi_found:
        print(f"  Warning: No 'sabi' label found in {annotation_path}. All labels will be 0.")

    # 抽出したMFCC (転置してフレームを行に) とラベルをリストに追加
    all_features.append(mfccs.T) # (フレーム数, n_mfcc) の形状にする
    all_labels.append(labels)
    song_names.append(song_name)

print("\n--- 特徴量抽出が完了しました ---")
print("処理された楽曲名:", song_names)

# 全楽曲の特徴量とラベルを結合 (Numpy配列として)
# Vstack: 垂直方向 (行方向) に配列を結合
X = np.vstack(all_features)
y = np.hstack(all_labels) # hstack: 水平方向 (列方向) に配列を結合 (1次元配列なのでそのまま結合)
for idx, feat in enumerate(all_features):
    print(f"{song_names[idx]} の特徴量 shape: {feat.shape}")
print(f"\n最終的な特徴量の形状 (X): {X.shape}")
print(f"最終的なラベルの形状 (y): {y.shape}")

# サビと非サビのフレーム数を確認 (クラスの不均衡度合いを見る)
num_sabi_frames = np.sum(y == 1)
num_non_sabi_frames = np.sum(y == 0)
print(f"サビのフレーム数 (ラベル1): {num_sabi_frames}")
print(f"非サビのフレーム数 (ラベル0): {num_non_sabi_frames}")
print(f"サビの割合: {num_sabi_frames / len(y) * 100:.2f}%")

# X (特徴量) と y (ラベル) が、モデル学習に使用するデータです。
# Xは (全フレーム数, MFCC次元数) の2次元配列
# yは (全フレーム数,) の1次元配列 (各フレームのラベル)

np.save('X.npy', X)
np.save('y.npy', y)
print("特徴量とラベルを 'X.npy', 'y.npy' として保存")