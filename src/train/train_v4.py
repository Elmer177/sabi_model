"""v4: ロジスティック回帰（特徴量の重み付き和でサビの確率を出す，最も単純なモデル）

特徴量は v3_2（これまでで最も良かった MFCC + 繰り返し 8 秒）と同じにして，モデルだけを変える
（v3_2 との差 = モデルの違いの効果）。
"""
import os
import sys
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量（v3_2 と同じ）
GROUPS = ["mfcc", "rep_all_32"]
# モデル
def make_model():
    return LogisticRegression(class_weight="balanced", max_iter=1000)
if __name__ == "__main__":
    run_experiment("v4", GROUPS, make_model)
