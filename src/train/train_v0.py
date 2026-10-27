"""v0: すべての区間を「非サビ」と予測するベースライン

特徴量を見ずに常に 0 を返す。学習するモデルがこれを超えられなければ，何も学べていないことになる。
"""
import os
import sys
from sklearn.dummy import DummyClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量
GROUPS = []
# モデル
def make_model():
    return DummyClassifier(strategy="constant", constant=0)
if __name__ == "__main__":
    run_experiment("v0", GROUPS, make_model)
