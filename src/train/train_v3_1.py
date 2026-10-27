"""v3_1: v3 から「繰り返し特徴量の作り方」だけ変更（repeat → rep_all_8）

繰り返しとみなす長さを 4 秒 → 2 秒に短くする（短いフレーズの繰り返しも拾う）。
ランダムフォレストの設定は v3（= v1_4）と同じ。
"""
import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量（v3 からの変更: repeat → rep_all_8）
GROUPS = ["mfcc", "rep_all_8"]
# モデル（v3 と同じ）
def make_model():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_features=None,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
if __name__ == "__main__":
    run_experiment("v3_1", GROUPS, make_model)
