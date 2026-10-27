"""学習用・最終テスト用のデータセットを作る

    学習用    : data/music_dataset/audio + annotation（20 曲）           → cache/dataset.pkl
    最終テスト: data/music_dataset/test_audio + test_annotation（3 曲） → cache/dataset_test.pkl

一度作れば，学習スクリプトはこのファイルを読むだけでよい（音声の読み込みは数分かかるため）。
すでにあるデータセットは作り直さない。曲・ラベル・特徴量の定義を変えたときは --force をつけて実行する:
    venv/bin/python src/features/build_dataset.py --force

学習用と同じ音声ファイル（中身が完全に一致するもの）は，最終テストから自動で除外する
（学習に使った曲でテストすると成績が高く出すぎるため）。

保存される中身: 曲ごとの辞書のリスト
    {"id": 曲番号, "X": 特徴量の DataFrame（区間 × 特徴量）, "y": ラベル（1 = サビ）, "times": 区間の開始時刻 [秒]}
"""

import hashlib
import os
import pickle
import sys

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from features import extract as fx
from paths import (ANNOTATION_DIR, AUDIO_DIR, DATASET_PATH, TEST_ANNOTATION_DIR, TEST_AUDIO_DIR,
                   TEST_DATASET_PATH)

NUM_SONGS = 20
NUM_TEST_SONGS = 3


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def build_songs(audio_dir, annotation_dir, ids):
    songs = []
    for i in ids:
        y_audio, _ = librosa.load(os.path.join(audio_dir, f"song_{i:03d}.wav"), sr=fx.SR)
        feats, times = fx.extract_features(y_audio)
        ann = pd.read_csv(os.path.join(annotation_dir, f"label_{i:03d}.txt"), sep="\t", header=None,
                          names=["start", "end", "label"])
        lab = fx.labels_for(times, ann)
        songs.append({"id": i, "X": feats, "y": lab, "times": times})
        print(f"  song_{i:03d}: {len(lab)} 区間（{times[-1]:.0f} 秒），サビ {lab.mean():.0%}")
    return songs


def save(songs, path):
    with open(path, "wb") as f:
        pickle.dump(songs, f)
    print(f"全 {sum(len(s['y']) for s in songs):,} 区間，サビの割合 "
          f"{np.mean(np.concatenate([s['y'] for s in songs])):.1%} → {os.path.relpath(path)} に保存")


def build(force=False):
    train_ids = list(range(1, NUM_SONGS + 1))
    if force or not os.path.exists(DATASET_PATH):
        print("学習用データセットを作成")
        save(build_songs(AUDIO_DIR, ANNOTATION_DIR, train_ids), DATASET_PATH)
    else:
        print(f"学習用データセットは作成済み（{os.path.relpath(DATASET_PATH)}）")

    if force or not os.path.exists(TEST_DATASET_PATH):
        train_hashes = {file_hash(os.path.join(AUDIO_DIR, f"song_{i:03d}.wav")): i for i in train_ids}
        test_ids = []
        for i in range(1, NUM_TEST_SONGS + 1):
            dup = train_hashes.get(file_hash(os.path.join(TEST_AUDIO_DIR, f"song_{i:03d}.wav")))
            if dup is not None:
                print(f"  注意: test_audio/song_{i:03d}.wav は学習用 audio/song_{dup:03d}.wav と同じファイル → テストから除外")
            else:
                test_ids.append(i)
        print(f"最終テスト用データセットを作成（{len(test_ids)} 曲）")
        save(build_songs(TEST_AUDIO_DIR, TEST_ANNOTATION_DIR, test_ids), TEST_DATASET_PATH)
    else:
        print(f"最終テスト用データセットは作成済み（{os.path.relpath(TEST_DATASET_PATH)}）")


if __name__ == "__main__":
    build(force="--force" in sys.argv)
