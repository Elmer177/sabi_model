"""学習・評価の共通処理
    1. cache/dataset.pkl を読み込む（features/build_dataset.py で作成済み）
    2. 指定した特徴量グループの列を選ぶ
    3. 曲単位の分割（1 曲を丸ごとテスト × 20 回）で学習・予測
    4. 予測を平滑化（≈4.3 秒の中央値フィルタ）し，生の予測と平滑化後の両方で正解率・サビの F1 を計算
       主な指標は「平滑化後の F1」（実際にサビ区間を出すときは必ず平滑化するため）
    5. 曲ごとの結果を results/result_<名前>.csv，予測確率を results/predictions/<名前>.npz に保存し，
       設定と平均成績を results/experiments.csv に 1 行追記する

最終テスト（data/music_dataset/test_*）は run_final_test で，モデルを決め終わった後に一度だけ使う。
"""

import datetime
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from features import extract as fx
from paths import (DATASET_PATH, EXPERIMENT_LOG, PREDICTIONS_DIR, RESULTS_DIR, ROOT_DIR,
                   TEST_DATASET_PATH)

FEATURE_GROUPS = fx.GROUPS  # "mfcc", "context", "energy", "chroma", "position", "repeat"
SMOOTH_SEGS = 17  # 平滑化の窓: 17 区間 ≈ 4.3 秒（全モデル共通で固定。変えると過去の実験と比べられなくなる）
THRESHOLD = 0.5  # サビと判定する確率のしきい値（全モデル共通）


def load_dataset(path=DATASET_PATH):
    """[{"id", "X"（DataFrame）, "y", "times"}, ...] を返す"""
    with open(path, "rb") as f:
        return pickle.load(f)

def describe_dataset(songs):
    print(f"{len(songs)} 曲，全 {sum(len(s['y']) for s in songs):,} 区間，"
          f"サビの割合 {np.mean(np.concatenate([s['y'] for s in songs])):.1%}")

def stack(song_list, cols):
    """曲のリスト → (X, y)。複数の曲の特徴量の表を縦につなげる"""
    X = pd.concat([s["X"][cols] for s in song_list]).to_numpy()
    y = np.concatenate([s["y"] for s in song_list])
    return X, y


def chorus_probability(model, X):
    """サビ（クラス 1）の確率。predict_proba がないモデルは 0/1 の予測をそのまま確率とみなす"""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    return model.predict(X).astype(float)


def smooth(prob):
    """確率を時間方向に中央値フィルタでならす（一瞬だけのサビ判定・サビの途中の短い穴を消す）"""
    return median_filter(prob, size=SMOOTH_SEGS)


def scores(y_true, prob):
    """生の予測と平滑化後の予測の正解率・F1"""
    raw = (prob >= THRESHOLD).astype(int)
    smoothed = (smooth(prob) >= THRESHOLD).astype(int)
    return {"F1(平滑化)": f1_score(y_true, smoothed, zero_division=0), "F1": f1_score(y_true, raw, zero_division=0),
            "正解率(平滑化)": accuracy_score(y_true, smoothed), "正解率": accuracy_score(y_true, raw)}


def fit_model(make_model, X, y, fit_kwargs=None):
    model = make_model()
    model.fit(X, y, **(fit_kwargs(X, y) if fit_kwargs else {}))
    return model


def leave_one_song_out(songs, make_model, cols, fit_kwargs=None):
    """曲単位の分割。曲ごとの成績の DataFrame と，曲ごとのサビ確率 {曲番号: 配列} を返す

    make_model : 引数なしで新しいモデルを返す関数
    fit_kwargs : 学習データ (X, y) を受け取り，fit に渡す追加引数の辞書を返す関数（sample_weight など。不要なら None）
    """
    rows, probs = [], {}
    for test in songs:
        X_tr, y_tr = stack([s for s in songs if s["id"] != test["id"]], cols)
        X_te, y_te = stack([test], cols)
        model = fit_model(make_model, X_tr, y_tr, fit_kwargs)
        prob = chorus_probability(model, X_te)
        probs[test["id"]] = prob
        rows.append({"曲": test["id"], **scores(y_te, prob)})
    return pd.DataFrame(rows), probs


def git_version():
    """今のコミット番号（未コミットの変更があれば末尾に +dirty）"""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT_DIR, capture_output=True,
                               text=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_experiment(row):
    """results/experiments.csv に 1 行追記（同じ名前の実験は最新の 1 行だけ残す）"""
    log = pd.read_csv(EXPERIMENT_LOG) if os.path.exists(EXPERIMENT_LOG) else pd.DataFrame()
    if len(log):
        log = log[~((log["名前"] == row["名前"]) & (log["評価"] == row["評価"]))]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(EXPERIMENT_LOG, index=False)


def print_summary(res):
    for col in ["F1(平滑化)", "F1", "正解率(平滑化)", "正解率"]:
        print(f"平均 {col:9s}: {res[col].mean():.3f}（最小 {res[col].min():.3f}，最大 {res[col].max():.3f}）")


def run_experiment(name, groups, make_model, fit_kwargs=None, verbose=True):
    """データ読み込み → 曲単位の評価 → 表示 → 保存と記録。(結果の表, サビ確率) を返す

    name       : 実験の名前（"v0" など。保存ファイル名に使う）
    groups     : 使う特徴量グループのリスト（FEATURE_GROUPS から選ぶ）
    make_model : 引数なしで新しいモデルを返す関数
    """
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"不明な特徴量グループ: {unknown}（選べるのは {FEATURE_GROUPS}）")

    songs = load_dataset()
    cols = fx.columns_for(groups)
    print(f"[{name}] 特徴量 {groups}（{len(cols)} 列），モデル {make_model()!r}")
    describe_dataset(songs)

    res, probs = leave_one_song_out(songs, make_model, cols, fit_kwargs)

    if verbose:
        print(res.round(3).to_string(index=False))
    print_summary(res)

    res.to_csv(os.path.join(RESULTS_DIR, f"result_{name}.csv"), index=False)
    np.savez_compressed(os.path.join(PREDICTIONS_DIR, f"{name}.npz"), **{str(k): v for k, v in probs.items()})
    log_experiment({
        "名前": name, "評価": "曲単位の分割（学習用 20 曲）", "日時": datetime.datetime.now().isoformat(timespec="seconds"),
        "コミット": git_version(), "特徴量": " ".join(groups), "列数": len(cols), "モデル": repr(make_model()),
        "fit_kwargs": fit_kwargs.__name__ if fit_kwargs else "", "平滑化区間": SMOOTH_SEGS, "しきい値": THRESHOLD,
        **{f"平均 {c}": res[c].mean() for c in ["F1(平滑化)", "F1", "正解率(平滑化)", "正解率"]},
        "F1(平滑化) 標準偏差": res["F1(平滑化)"].std(),
    })
    print(f"results/result_{name}.csv，results/predictions/{name}.npz に保存，results/experiments.csv に記録")
    return res, probs


def run_final_test(name, groups, make_model, fit_kwargs=None):
    """最終テスト: 学習用 20 曲すべてで学習し，最終テスト用の曲（学習に使っていない曲）で評価する

    モデルの比較・設定選びには使わないこと。設定を決め終わった最終モデルで一度だけ実行する。
    （何度も見て設定を変えると，最終テストの曲に合わせた設定を選ぶことになり，公平な評価でなくなる）
    """
    if not os.path.exists(TEST_DATASET_PATH):
        raise FileNotFoundError("最終テスト用データセットがない。先に features/build_dataset.py を実行する")
    train, test = load_dataset(), load_dataset(TEST_DATASET_PATH)
    cols = fx.columns_for(groups)
    X_tr, y_tr = stack(train, cols)
    model = fit_model(make_model, X_tr, y_tr, fit_kwargs)
    rows = []
    for s in test:
        X_te, y_te = stack([s], cols)
        rows.append({"曲": s["id"], **scores(y_te, chorus_probability(model, X_te))})
    res = pd.DataFrame(rows)
    print(f"[{name}] 最終テスト（{len(test)} 曲）")
    print(res.round(3).to_string(index=False))
    print_summary(res)
    res.to_csv(os.path.join(RESULTS_DIR, f"final_test_{name}.csv"), index=False)
    log_experiment({
        "名前": name, "評価": f"最終テスト（{len(test)} 曲）", "日時": datetime.datetime.now().isoformat(timespec="seconds"),
        "コミット": git_version(), "特徴量": " ".join(groups), "列数": len(cols), "モデル": repr(make_model()),
        "fit_kwargs": fit_kwargs.__name__ if fit_kwargs else "", "平滑化区間": SMOOTH_SEGS, "しきい値": THRESHOLD,
        **{f"平均 {c}": res[c].mean() for c in ["F1(平滑化)", "F1", "正解率(平滑化)", "正解率"]},
        "F1(平滑化) 標準偏差": res["F1(平滑化)"].std(),
    })
    return res, model
