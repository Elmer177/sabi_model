"""v1: MFCC + ランダムフォレスト（元のモデル sabi_model.py と同じ組み合わせ）

元のモデルと同じ特徴量の種類・同じモデル設定を，今の前処理（≈0.25 秒区間・曲ごとの標準化）と
今の評価（曲単位の分割・平滑化後の F1）で測り直す。v1 系列（v1_1, v1_2, …）の出発点。
"""
import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量
GROUPS = ["mfcc"]
# モデル（元のモデルと同じ設定。n_jobs=-1 は並列計算するだけで結果は変わらない）
def make_model():
    return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
if __name__ == "__main__":
    run_experiment("v1", GROUPS, make_model)
