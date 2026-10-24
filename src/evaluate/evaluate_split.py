"""ステップ1｜評価方法の見直し: フレーム単位のランダム分割 vs 曲単位の分割

sabi_model.py は train_test_split でフレームをランダムに分けている。
隣り合うフレーム（約 23 ms 違い）はほぼ同じ音なので，同じ曲の同じサビが学習とテストの両方に入り，
精度が高く出すぎている可能性がある。ここでは同じモデル・同じ特徴量（MFCC 13 次元）のまま
  (A) フレーム単位のランダム分割（sabi_model.py と同じ）
  (B) 曲単位の分割（1 曲を丸ごとテストに回し，残り 19 曲で学習。20 回くり返す）
を比べる。

既存の X.npy / y.npy / 学習済みモデルは上書きしない。
曲ごとの特徴量は cache/features_per_song.npz にキャッシュする（2 回目以降は音声を読み直さない）。
結果は results/result_split_comparison.csv。
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from paths import AUDIO_DIR, ANNOTATION_DIR, CACHE_DIR, MODELS_DIR, RESULTS_DIR

ANNOTATIONS_DIR = ANNOTATION_DIR
NUM_SONGS = 20
with open(os.path.join(MODELS_DIR, "feature_extraction_config.json")) as f:
    config = json.load(f)
SR, N_MFCC, HOP_LENGTH = config["SR"], config["N_MFCC"], config["HOP_LENGTH"]
CACHE = os.path.join(CACHE_DIR, "features_per_song.npz")
RESULT_CSV = os.path.join(RESULTS_DIR, "result_split_comparison.csv")
STRIDE = 4  # 学習・評価に使うフレームを 4 つに 1 つに間引く（隣接フレームはほぼ同じなので情報はほぼ減らない。計算を 1/4 に）


# ============================================================
# 1. 曲ごとに特徴量とラベルを作る（feature_analysis.py と同じ処理 + 曲番号を残す）
# ============================================================
def extract_song(i):
    song = f"song_{i:03d}"
    y_audio, sr = librosa.load(os.path.join(AUDIO_DIR, f"{song}.wav"), sr=SR)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=2048)
    ann = pd.read_csv(os.path.join(ANNOTATIONS_DIR, f"label_{i:03d}.txt"), sep="\t", header=None,
                      names=["start", "end", "label"])
    labels = np.zeros(mfcc.shape[1], dtype=int)
    for _, row in ann.iterrows():
        if str(row["label"]).strip().lower() == "sabi":
            s = max(0, librosa.time_to_frames(row["start"], sr=sr, hop_length=HOP_LENGTH))
            e = min(mfcc.shape[1], librosa.time_to_frames(row["end"], sr=sr, hop_length=HOP_LENGTH))
            labels[s:e] = 1
    return mfcc.T, labels


if os.path.exists(CACHE):
    cache = np.load(CACHE)
    X, y, song_id = cache["X"], cache["y"], cache["song_id"]
    print(f"キャッシュ {CACHE} を読み込み")
else:
    feats, labs, ids = [], [], []
    for i in range(1, NUM_SONGS + 1):
        f_, l_ = extract_song(i)
        feats.append(f_), labs.append(l_), ids.append(np.full(len(l_), i))
        print(f"  song_{i:03d}: {len(l_)} フレーム，サビ {l_.mean():.0%}")
    X, y, song_id = np.vstack(feats), np.concatenate(labs), np.concatenate(ids)
    np.savez_compressed(CACHE, X=X, y=y, song_id=song_id)

# 既存の X.npy / y.npy と一致するか確認（同じ処理で作れているか）
if os.path.exists(os.path.join(CACHE_DIR, "X.npy")):
    X_old, y_old = np.load(os.path.join(CACHE_DIR, "X.npy")), np.load(os.path.join(CACHE_DIR, "y.npy"))
    same = X_old.shape == X.shape and np.allclose(X_old, X, atol=1e-3) and np.array_equal(y_old, y)
    print(f"既存の X.npy / y.npy と一致: {same}")

X, y, song_id = X[::STRIDE], y[::STRIDE], song_id[::STRIDE]
print(f"使用フレーム数: {len(y):,}（{STRIDE} つに 1 つ），曲数: {len(np.unique(song_id))}，サビの割合: {y.mean():.1%}")


def make_model():
    # sabi_model.py と同じ設定（n_jobs=-1 で並列化だけ追加）
    return RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)


# ============================================================
# 2. (A) フレーム単位のランダム分割（sabi_model.py と同じ方法）
# ============================================================
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pred = make_model().fit(X_tr, y_tr).predict(X_te)
acc_a, f1_a = accuracy_score(y_te, pred), f1_score(y_te, pred)
print(f"\n(A) フレーム単位のランダム分割: 正解率 {acc_a:.3f}，サビの F1 {f1_a:.3f}")

# ============================================================
# 3. (B) 曲単位の分割（Leave-One-Song-Out）
# ============================================================
rows = []
for s in np.unique(song_id):
    test = song_id == s
    pred = make_model().fit(X[~test], y[~test]).predict(X[test])
    rows.append({"曲": f"song_{s:03d}", "サビの割合": y[test].mean(), "正解率": accuracy_score(y[test], pred),
                 "サビの F1": f1_score(y[test], pred, zero_division=0), "サビと予測した割合": pred.mean()})
    print(f"  {rows[-1]['曲']}: 正解率 {rows[-1]['正解率']:.3f}，F1 {rows[-1]['サビの F1']:.3f}")
res = pd.DataFrame(rows)

print(f"\n(B) 曲単位の分割（20 曲の平均）: 正解率 {res['正解率'].mean():.3f}，サビの F1 {res['サビの F1'].mean():.3f}")
print(f"    参考: 全フレームを「非サビ」と答えるだけの正解率 = {1 - y.mean():.3f}")
print(f"\n(A) − (B): 正解率 {acc_a - res['正解率'].mean():+.3f}，F1 {f1_a - res['サビの F1'].mean():+.3f}")
res.to_csv(RESULT_CSV, index=False)
print("曲ごとの結果を result_split_comparison.csv に保存")
