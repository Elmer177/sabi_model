"""実験どうしを比べる

results/experiments.csv（設定と平均）と results/result_<名前>.csv（曲ごとの成績）を読み，
主な指標「平滑化後の F1」で一覧表と図を作る。曲が 20 曲と少ないので，平均の差だけでなく
基準の実験と比べて「何曲で良くなったか」と，曲ごとの対応を取った検定（ウィルコクソン符号順位検定）も出す。

使い方:
    venv/bin/python src/evaluate/compare.py                 # 全実験を v0 と比べる
    venv/bin/python src/evaluate/compare.py --base v1       # 基準を v1 にする
    venv/bin/python src/evaluate/compare.py v1 v2 v3        # 比べる実験を選ぶ

出力: 画面の表，results/comparison.csv，results/figures/compare.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from paths import EXPERIMENT_LOG, FIGURES_DIR, RESULTS_DIR

plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "sans-serif"]
METRIC = "F1(平滑化)"  # 主な指標
CV_LABEL = "曲単位の分割（学習用 20 曲）"


def paired_test(diff):
    """曲ごとの差のウィルコクソン符号順位検定（差がすべて 0 なら検定できないので NaN）"""
    if np.allclose(diff, 0):
        return np.nan
    return stats.wilcoxon(diff).pvalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="比べる実験の名前（省略すると全部）")
    parser.add_argument("--base", default="v0", help="基準にする実験の名前（既定: v0）")
    args = parser.parse_args()

    log = pd.read_csv(EXPERIMENT_LOG)
    log = log[log["評価"] == CV_LABEL]
    if args.names:
        log = log[log["名前"].isin(args.names + [args.base])]
    per_song = {n: pd.read_csv(os.path.join(RESULTS_DIR, f"result_{n}.csv")).set_index("曲") for n in log["名前"]}
    if args.base not in per_song:
        sys.exit(f"基準の実験 {args.base} の結果がない（results/result_{args.base}.csv）")
    base = per_song[args.base][METRIC]

    rows = []
    for _, r in log.iterrows():
        diff = per_song[r["名前"]][METRIC] - base
        rows.append({
            "名前": r["名前"], "特徴量": r["特徴量"], "モデル": r["モデル"][:40],
            METRIC: r[f"平均 {METRIC}"], "F1(生)": r["平均 F1"], "正解率(平滑化)": r[f"平均 正解率(平滑化)"],
            f"{args.base} との差": diff.mean(),
            "良くなった曲": f"{(diff > 0).sum()}/{len(diff)}",
            "p 値": paired_test(diff) if r["名前"] != args.base else np.nan,
            "コミット": r["コミット"],
        })
    table = pd.DataFrame(rows).sort_values(METRIC, ascending=False)
    pd.set_option("display.width", 200)
    print(f"主な指標: {METRIC}（20 曲の平均）。基準: {args.base}。p 値はウィルコクソン符号順位検定")
    print(table.round(3).to_string(index=False))
    table.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)

    # 図: 実験ごとの平均（棒）と曲ごとの値（点）
    order = table["名前"].tolist()[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(order) + 1.5))
    rng = np.random.default_rng(0)
    for y_pos, name in enumerate(order):
        vals = per_song[name][METRIC].to_numpy()
        ax.barh(y_pos, vals.mean(), color="0.6" if name == args.base else "C0")
        ax.plot(vals, y_pos + rng.uniform(-0.25, 0.25, len(vals)), ".", color="k", ms=3, alpha=0.5)
    ax.set_yticks(range(len(order)), order)
    ax.set(xlim=(0, 1), xlabel=f"サビの {METRIC}（曲単位の分割。棒 = 平均，点 = 各曲）", title="実験の比較")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "compare.png"), dpi=150)
    print(f"\nresults/comparison.csv，results/figures/compare.png に保存")


if __name__ == "__main__":
    main()
