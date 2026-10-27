"""v3_6: v3 から繰り返しを 4 つの長さすべてに

短い繰り返し（フレーズ）から長い繰り返し（サビ全体）まで，長さの違う繰り返しを同時に使う。
ランダムフォレストの設定は v3（= v1_4）と同じ。
"""
import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量
GROUPS = ["mfcc", "rep_all_8", "rep_all_16", "rep_all_32", "rep_all_64"]
# モデル（v3 と同じ）
def make_model():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_features=None,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
if __name__ == "__main__":
    run_experiment("v3_6", GROUPS, make_model)
