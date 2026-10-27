"""v7: ニューラルネットワーク（隠れ層 64・32 の多層パーセプトロン）

特徴量は v3_2（これまでで最も良かった MFCC + 繰り返し 8 秒）と同じにして，モデルだけを変える
（v3_2 との差 = モデルの違いの効果）。
"""
import os
import sys
from sklearn.neural_network import MLPClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量（v3_2 と同じ）
GROUPS = ["mfcc", "rep_all_32"]
# モデル
def make_model():
    return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=300, random_state=42)
if __name__ == "__main__":
    run_experiment("v7", GROUPS, make_model)
