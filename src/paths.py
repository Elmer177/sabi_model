"""プロジェクト共通のパス

各スクリプトは先頭で src/ を import パスに追加してから `from paths import ...` で使う:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
    from paths import AUDIO_DIR, MODELS_DIR, ...

フォルダ構成
    data/music_dataset/   音源（audio, test_audio）とサビのラベル（annotation, test_annotation）
    my_wav_files/         YouTube から取ってきた音源（src/prep/linktowav.py の出力先）
    src/prep/             データ準備（音源のダウンロード）
    src/features/         特徴量の定義（extract.py）とデータセットの作成（build_dataset.py）
    src/common/           学習・評価の共通処理（experiment.py）
    src/train/            学習（モデルごとに 1 ファイル。特徴量とモデルだけを書く）
    src/evaluate/         評価
    src/detect/           学習済みモデルでサビ検出
    models/               学習済みモデルと特徴量の設定
    cache/                特徴量などの中間ファイル（作り直せる）
    results/              評価結果の CSV と図（results/figures/）
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data", "music_dataset")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotation")
TEST_AUDIO_DIR = os.path.join(DATA_DIR, "test_audio")
TEST_ANNOTATION_DIR = os.path.join(DATA_DIR, "test_annotation")
MY_WAV_DIR = os.path.join(ROOT_DIR, "my_wav_files")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

DATASET_PATH = os.path.join(CACHE_DIR, "dataset.pkl")  # features/build_dataset.py が作る学習用データセット（20 曲）
TEST_DATASET_PATH = os.path.join(CACHE_DIR, "dataset_test.pkl")  # 同じく最終テスト用（モデル選びには使わない）
EXPERIMENT_LOG = os.path.join(RESULTS_DIR, "experiments.csv")  # 実験ごとの設定と平均成績の記録
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")  # 実験ごとの予測確率（後から評価し直すため）

for _d in (MODELS_DIR, CACHE_DIR, RESULTS_DIR, FIGURES_DIR, PREDICTIONS_DIR):
    os.makedirs(_d, exist_ok=True)
