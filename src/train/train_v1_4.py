"""v1_4: v1_1 から「各分岐で候補にする特徴量の数」だけ変更（√13 ≈ 3 個 → 13 個すべて）

木の本数は 100（v1_1: 0.770）と 500（v1_3: 0.768）で差がなかったので 100 のまま。
既定では各分岐で 13 個中ランダムな 3 個から選ぶ。すべての MFCC を候補にして 1 本 1 本の木を強くする。
"""
import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量
GROUPS = ["mfcc"]
# モデル（v1_1 からの変更: max_features="sqrt" → None（すべて））
def make_model():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_features=None, class_weight="balanced",
                                  random_state=42, n_jobs=-1)
if __name__ == "__main__":
    run_experiment("v1_4", GROUPS, make_model)
