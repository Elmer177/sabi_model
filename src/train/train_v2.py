"""サビ検出 v2 の学習（ひな形）

下準備（済み）
    - 特徴量の定義: src/features/features_v2.py
    - データセット:  cache/dataset_v2.pkl（src/features/build_dataset_v2.py で作成。20 曲，≈0.25 秒ごとの区間）

このファイルで決めること（TODO の部分）
    1. 使う特徴量グループ
    2. モデルとその設定
    3. 評価の方法（下に曲単位の分割の関数を用意してある）
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from features import features_v2 as fv
from paths import CACHE_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR  # noqa: F401  （保存先として使う）

# ============================================================
# データセットの読み込み
# ============================================================
with open(os.path.join(CACHE_DIR, "dataset_v2.pkl"), "rb") as f:
    songs = pickle.load(f)  # [{"id", "X"（DataFrame）, "y", "times"}, ...] × 20 曲

print(f"{len(songs)} 曲，全 {sum(len(s['y']) for s in songs):,} 区間，"
      f"サビの割合 {np.mean(np.concatenate([s['y'] for s in songs])):.1%}")
print("特徴量グループと列数:")
for g in ["mfcc", "energy", "chroma", "position", "repeat"]:
    print(f"  {g:9s}: {len(fv.columns_for([g]))} 列")
print("  context  : 一緒に使う mfcc / energy / chroma の各列に，前後 ≈4 秒の平均・標準偏差を追加（列数 × 2）")


# ============================================================
# 道具（必要に応じて使う）
# ============================================================
def stack(song_list, cols):
    """曲のリスト → (X, y)。cols は使う列名のリスト（fv.columns_for(グループのリスト) で作れる）"""
    X = pd.concat([s["X"][cols] for s in song_list]).to_numpy()
    y = np.concatenate([s["y"] for s in song_list])
    return X, y


def leave_one_song_out(make_model, cols, fit_kwargs=None):
    """曲単位の分割（1 曲を丸ごとテスト × 曲数）。曲ごとの正解率・F1 の DataFrame と，曲ごとの予測を返す

    make_model : 引数なしで新しいモデルを返す関数（例: lambda: RandomForestClassifier(...)）
    fit_kwargs : 学習データ (X, y) を受け取り，fit に渡す追加引数の辞書を返す関数（sample_weight など。不要なら None）
    """
    rows, preds = [], {}
    for test in songs:
        X_tr, y_tr = stack([s for s in songs if s["id"] != test["id"]], cols)
        X_te, y_te = stack([test], cols)
        model = make_model()
        model.fit(X_tr, y_tr, **(fit_kwargs(X_tr, y_tr) if fit_kwargs else {}))
        pred = model.predict(X_te)
        preds[test["id"]] = pred
        rows.append({"曲": test["id"], "正解率": accuracy_score(y_te, pred), "F1": f1_score(y_te, pred, zero_division=0)})
    return pd.DataFrame(rows), preds


# ============================================================
# TODO 1: 使う特徴量を決める
#   グループ: "mfcc", "context", "energy", "chroma", "position", "repeat"
# ============================================================
groups = ["mfcc"]  # 例
cols = fv.columns_for(groups)
print(f"\n使う特徴量: {groups}（{len(cols)} 列）")

# ============================================================
# TODO 2: モデルを決める
# ============================================================
# 例:
# from sklearn.ensemble import RandomForestClassifier
# def make_model():
#     return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)

# ============================================================
# TODO 3: 評価する（曲単位の分割の例）
# ============================================================
# res, preds = leave_one_song_out(make_model, cols)
# print(res.round(3).to_string(index=False))
# print(f"平均: 正解率 {res['正解率'].mean():.3f}，F1 {res['F1'].mean():.3f}")
