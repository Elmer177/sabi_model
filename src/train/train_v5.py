"""v5: 勾配ブースティング（前の木の間違いを直すように木を 1 本ずつ足していく）

特徴量は v3_2（これまでで最も良かった MFCC + 繰り返し 8 秒）と同じにして，モデルだけを変える
（v3_2 との差 = モデルの違いの効果）。
"""
import os
import sys
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量（v3_2 と同じ）
GROUPS = ["mfcc", "rep_all_32"]
# モデル
def make_model():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, min_samples_leaf=20, class_weight="balanced", random_state=42)
if __name__ == "__main__":
    run_experiment("v5", GROUPS, make_model)
