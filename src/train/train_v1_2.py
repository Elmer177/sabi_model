"""v1_2: v1_1 から「葉の最小区間数」だけ変更（20 → 100）

v1_1 で学習しすぎを抑えると平滑化後の F1 が 0.664 → 0.770 に上がった。同じ方向にさらに強める。
葉に 100 区間（≈25 秒分）以上を必ず含める。
"""
import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量
GROUPS = ["mfcc"]
# モデル（v1_1 からの変更: min_samples_leaf=20 → 100）
def make_model():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=100, class_weight="balanced",
                                  random_state=42, n_jobs=-1)
if __name__ == "__main__":
    run_experiment("v1_2", GROUPS, make_model)
