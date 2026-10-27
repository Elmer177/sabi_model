"""v2: MFCC + 前後文脈 + ランダムフォレスト

v1 系列（MFCC のみ）は平滑化後の F1 ≈ 0.78 で頭打ちだった。MFCC はその瞬間（≈0.25 秒）の音色しか見ていない。
前後 ≈4 秒の MFCC の平均・標準偏差を加えて，「区間」としての音色を見る。
ランダムフォレストの設定は v1 系列で最も良かった v1_4 と同じにする（v1_4 との差 = 前後文脈の効果）。
"""
import os
import sys
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from common.experiment import run_experiment

# 使う特徴量（v1_4 からの変更: "context" を追加 → MFCC 13 列 + その前後平均・標準偏差 26 列）
GROUPS = ["mfcc", "context"]
# モデル（v1_4 と同じ）
def make_model():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_features=None,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
if __name__ == "__main__":
    run_experiment("v2", GROUPS, make_model)
