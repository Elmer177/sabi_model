"""v6: k 近傍法（学習データの中で特徴量が最も近い 101 区間の多数決）

特徴量は v3_2（これまでで最も良かった MFCC + 繰り返し 8 秒）と同じにして，モデルだけを変える
（v3_2 との差 = モデルの違いの効果）。
"""
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量（v3_2 と同じ）
GROUPS = ["mfcc", "rep_all_32"]
# モデル
def make_model():
    return KNeighborsClassifier(n_neighbors=101, n_jobs=-1)
if __name__ == "__main__":
    run_experiment("v6", GROUPS, make_model)
