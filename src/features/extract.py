"""特徴量の定義（1 曲の音声 → 区間ごとの特徴量）

旧モデル（train/sabi_model.py）は 23 ms ごとの MFCC 13 次元だけだった。
ここでは 1 区間 ≈ 0.25 秒ごとに，次の特徴量グループを作る（train ではこの中から使うグループを選ぶ）:

  mfcc     : 音色（MFCC 13 次元）
  context  : 前後 ≈ 4 秒の平均・標準偏差（一瞬ではなく「区間」として見る）
  energy   : 音量（RMS）とアタックの強さ（onset strength）＝ 盛り上がり
  chroma   : 12 音の高さの分布（和音・メロディ）
  position : 曲の中の位置（0 = 冒頭，1 = 終わり）
  repeat   : 曲の中で何度も繰り返される区間か（自己類似度行列から計算）

音量や音質は曲ごとの録音・マスタリングで違うので，position 以外は曲ごとに標準化（平均 0，標準偏差 1）する。
"""

import librosa
import numpy as np
import pandas as pd

SR = 22050
HOP_LENGTH = 512  # 1 フレーム ≈ 23.2 ms
SEG_FRAMES = 11  # 11 フレームを平均して 1 区間 ≈ 0.255 秒
SEG_SEC = SEG_FRAMES * HOP_LENGTH / SR
CONTEXT_SEGS = 16  # 前後文脈の窓 ≈ 4 秒
REPEAT_SMOOTH_SEGS = 16  # 繰り返しを判定する長さ ≈ 4 秒
REPEAT_MIN_LAG_SEGS = 40  # 10 秒以上離れた区間との繰り返しだけ数える（すぐ隣の自分自身を除く）

GROUPS = ["mfcc", "context", "energy", "chroma", "position", "repeat"]


def _segment_mean(frames):
    """(次元, フレーム数) → (区間数, 次元)。SEG_FRAMES フレームずつ平均する"""
    n_seg = frames.shape[1] // SEG_FRAMES
    return frames[:, : n_seg * SEG_FRAMES].reshape(frames.shape[0], n_seg, SEG_FRAMES).mean(axis=2).T


def _zscore(a):
    return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-8)


def repetition_scores(seg_feats):
    """各区間が「曲の別の場所で繰り返されているか」を表す 2 つの値を返す

    1. 区間どうしのコサイン類似度行列 S（N × N）を作る（第 2 回の相関と同じ考え方）
    2. 同じフレーズが繰り返されると，S の斜め（i と i+L）に高い値が並ぶ
       → 斜め方向に ≈4 秒の移動平均をとり，「4 秒以上続けて似ている」ところだけ高くする
    3. 区間 i について，10 秒以上離れたすべてのずれ L のうち最大値と上位 3 つの平均を取る
    """
    z = _zscore(seg_feats)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    S = z @ z.T
    n = len(S)
    kernel = np.ones(REPEAT_SMOOTH_SEGS) / REPEAT_SMOOTH_SEGS
    best = np.full((n, 3), -1.0)  # 各区間の上位 3 つの類似度
    # 重なりが平滑化の長さ（≈4 秒）より短いずれは判定できないので除く
    for lag in range(REPEAT_MIN_LAG_SEGS, n - REPEAT_SMOOTH_SEGS + 1):
        diag = np.convolve(np.diagonal(S, lag), kernel, mode="same")
        for idx in (np.arange(n - lag), np.arange(lag, n)):  # 組 (i, i+lag) の両方の区間に記録
            cand = np.column_stack([best[idx], diag])
            best[idx] = -np.sort(-cand, axis=1)[:, :3]
    return np.column_stack([best[:, 0], best.mean(axis=1)])


def extract_features(y_audio):
    """1 曲の音声 → (区間ごとの特徴量 DataFrame, 区間の開始時刻 [秒])"""
    mfcc = _segment_mean(librosa.feature.mfcc(y=y_audio, sr=SR, n_mfcc=13, hop_length=HOP_LENGTH, n_fft=2048))
    chroma = _segment_mean(librosa.feature.chroma_stft(y=y_audio, sr=SR, hop_length=HOP_LENGTH, n_fft=2048))
    rms = _segment_mean(librosa.feature.rms(y=y_audio, hop_length=HOP_LENGTH, frame_length=2048))
    onset = _segment_mean(librosa.onset.onset_strength(y=y_audio, sr=SR, hop_length=HOP_LENGTH)[None, :])
    n = min(len(mfcc), len(chroma), len(rms), len(onset))
    mfcc, chroma, rms, onset = mfcc[:n], chroma[:n], rms[:n], onset[:n]

    cols = {}
    for i in range(13):
        cols[f"mfcc_{i}"] = mfcc[:, i]
    cols["rms"] = np.log(rms[:, 0] + 1e-6)
    cols["onset"] = onset[:, 0]
    for i in range(12):
        cols[f"chroma_{i}"] = chroma[:, i]
    df = pd.DataFrame(cols)

    # 前後文脈: 中心 ≈4 秒の移動平均・移動標準偏差（MFCC・音量・クロマ）
    roll = df.rolling(CONTEXT_SEGS, center=True, min_periods=1)
    ctx = pd.concat([roll.mean().add_suffix("_ctxmean"), roll.std().fillna(0).add_suffix("_ctxstd")], axis=1)

    rep = repetition_scores(np.hstack([mfcc, chroma]))
    df = pd.concat([df, ctx], axis=1)
    df["repeat_max"], df["repeat_top3"] = rep[:, 0], rep[:, 1]

    df = (df - df.mean()) / (df.std() + 1e-8)  # 曲ごとの標準化
    df["position"] = np.arange(n) / max(n - 1, 1)  # 位置は標準化しない
    times = np.arange(n) * SEG_SEC
    return df, times


def columns_for(groups):
    """特徴量グループ名のリスト → 使う列名のリスト"""
    base_mfcc = [f"mfcc_{i}" for i in range(13)]
    base_energy = ["rms", "onset"]
    base_chroma = [f"chroma_{i}" for i in range(12)]
    cols = []
    if "mfcc" in groups:
        cols += base_mfcc
    if "energy" in groups:
        cols += base_energy
    if "chroma" in groups:
        cols += base_chroma
    if "context" in groups:  # 文脈は，使っている基本特徴量に対してだけ付ける
        base = (base_mfcc if "mfcc" in groups else []) + (base_energy if "energy" in groups else []) \
            + (base_chroma if "chroma" in groups else [])
        cols += [c + s for c in base for s in ("_ctxmean", "_ctxstd")]
    if "position" in groups:
        cols += ["position"]
    if "repeat" in groups:
        cols += ["repeat_max", "repeat_top3"]
    return cols


def labels_for(times, annotation_df):
    """区間の開始時刻とアノテーション（start, end, label）→ 区間ごとのラベル（区間の中心がサビなら 1）"""
    centers = times + SEG_SEC / 2
    lab = np.zeros(len(times), dtype=int)
    for _, row in annotation_df.iterrows():
        if str(row["label"]).strip().lower() == "sabi":
            lab[(centers >= row["start"]) & (centers < row["end"])] = 1
    return lab
