"""v1_5: v1_4 から「各木が学習に使う区間の割合」だけ変更（100% → 10%）

v1_4（各分岐で全特徴量を候補）で 0.785 に上がったので v1_4 を基準にする。
隣り合う区間はほぼ同じ音なので，1 本の木に全区間を見せる必要は薄い。各木に 10% だけ見せて木どうしを多様にする。
"""
import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量
GROUPS = ["mfcc"]
# モデル（v1_4 からの変更: max_samples=None → 0.1）
def make_model():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_features=None, max_samples=0.1,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
if __name__ == "__main__":
    run_experiment("v1_5", GROUPS, make_model)
