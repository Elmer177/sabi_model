"""v1_3: v1_1 から「木の本数」だけ変更（100 → 500）

葉の最小区間数は 20（v1_1: 0.770）と 100（v1_2: 0.768）で差がなかったので 20 に戻す。
木を増やすと多数決の確率のばらつきが減り，平滑化のしきい値付近での揺れが小さくなることを期待する。
"""
import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量
GROUPS = ["mfcc"]
# モデル（v1_1 からの変更: n_estimators=100 → 500）
def make_model():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=20, class_weight="balanced",
                                  random_state=42, n_jobs=-1)
if __name__ == "__main__":
    run_experiment("v1_3", GROUPS, make_model)
