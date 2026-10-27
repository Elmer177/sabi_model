"""v1_1: v1 から「葉の最小区間数」だけ変更（min_samples_leaf=1 → 20）

v1 では木を最後まで伸ばすため，サビの確率がとがって不安定になり，平滑化で消える曲があった（song_004 の F1 が 0）。
葉に 20 区間（≈5 秒分）以上を必ず含めて，木の学習しすぎを抑え，確率をなめらかにする。
"""
import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量
GROUPS = ["mfcc"]
# モデル（v1 からの変更: min_samples_leaf=20）
def make_model():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=20, class_weight="balanced",
                                  random_state=42, n_jobs=-1)
if __name__ == "__main__":
    run_experiment("v1_1", GROUPS, make_model)
