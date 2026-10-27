"""v3: MFCC + 繰り返し + ランダムフォレスト

v2 で前後文脈を加えても，平滑化後の F1 は v1_4 とほぼ同じだった（前後文脈と平滑化が同じ役割をしていた）。
そこで前後文脈は使わず，v1_4 に「曲の中で何度も繰り返される区間か」（自己類似度行列から計算した 2 列）だけを加える。
ランダムフォレストの設定は v1_4 と同じ（v1_4 との差 = 繰り返しの効果）。
"""
import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量（v1_4 からの変更: "repeat" を追加 → MFCC 13 列 + 繰り返し 2 列）
GROUPS = ["mfcc", "repeat"]
# モデル（v1_4 と同じ）
def make_model():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_features=None,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
if __name__ == "__main__":
    run_experiment("v3", GROUPS, make_model)
