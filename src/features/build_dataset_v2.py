"""学習用データセット（v2 特徴量）を作る

data/music_dataset の 20 曲から features_v2.py の特徴量とサビのラベルを作り，cache/dataset_v2.pkl に保存する。
一度作れば，学習スクリプトはこのファイルを読むだけでよい（音声の読み込みは数分かかるため）。

保存される中身: 曲ごとの辞書のリスト
    {"id": 曲番号, "X": 特徴量の DataFrame（区間 × 特徴量）, "y": ラベル（1 = サビ）, "times": 区間の開始時刻 [秒]}
"""

import os
import pickle
import sys

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from features import features_v2 as fv
from paths import ANNOTATION_DIR, AUDIO_DIR, CACHE_DIR

NUM_SONGS = 20
DATASET_PATH = os.path.join(CACHE_DIR, "dataset_v2.pkl")


def build():
    songs = []
    for i in range(1, NUM_SONGS + 1):
        y_audio, _ = librosa.load(os.path.join(AUDIO_DIR, f"song_{i:03d}.wav"), sr=fv.SR)
        feats, times = fv.extract_features(y_audio)
        ann = pd.read_csv(os.path.join(ANNOTATION_DIR, f"label_{i:03d}.txt"), sep="\t", header=None,
                          names=["start", "end", "label"])
        lab = fv.labels_for(times, ann)
        songs.append({"id": i, "X": feats, "y": lab, "times": times})
        print(f"  song_{i:03d}: {len(lab)} 区間（{times[-1]:.0f} 秒），サビ {lab.mean():.0%}")
    with open(DATASET_PATH, "wb") as f:
        pickle.dump(songs, f)
    print(f"全 {sum(len(s['y']) for s in songs):,} 区間，サビの割合 {np.mean(np.concatenate([s['y'] for s in songs])):.1%}")
    print(f"{DATASET_PATH} に保存")


if __name__ == "__main__":
    build()
