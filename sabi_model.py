from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib
plt.rcParams['font.family'] = 'AppleGothic'


DATASET_ROOT = 'music_dataset'
AUDIO_DIR = os.path.join(DATASET_ROOT, 'audio')
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, 'annotation')
NUM_SONGS = 20 
SR = 22050
N_MFCC = 13
HOP_LENGTH = 512
X = np.load('X.npy')
y = np.load('y.npy')

#バリデーション
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"訓練データの形状 (X_train): {X_train.shape}")
print(f"テストデータの形状 (X_test): {X_test.shape}")
print(f"訓練ラベルの形状 (y_train): {y_train.shape}")
print(f"テストラベルの形状 (y_test): {y_test.shape}")

#モデルの選択
print("\n学習開始")
# n_estimators: 決定木の数 
# random_state: シード値
# class_weight: クラス不均衡
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print("学習完了")

#評価
print("\n評価開始")
y_pred = model.predict(X_test)

print("\n分類レポート")
print(classification_report(y_test, y_pred, target_names=['Non-Chorus', 'Chorus']))

print("\n混同行列")
cm = confusion_matrix(y_test, y_pred)
print(cm)

#混同行列の可視化
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['予測:非サビ', '予測:サビ'], yticklabels=['正解:非サビ', '正解:サビ'])
plt.title('混同行列')
plt.xlabel('予測ラベル')
plt.ylabel('正解のラベル')
#plt.show()

# sabi_model.py のモデル評価セクションの後に以下を追加

print("\n特徴量の重要度")
# 各特徴量の重要度
importances = model.feature_importances_
feature_names = [f'MFCC {i}' for i in range(N_MFCC)]
sorted_indices = np.argsort(importances)[::-1] 
for i in sorted_indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# 特徴量の重要度の可視化
plt.figure(figsize=(10, 6))
plt.bar(range(N_MFCC), importances[sorted_indices], tick_label=[feature_names[i] for i in sorted_indices])
plt.xlabel('MFCC次元')
plt.ylabel('重要度')
plt.title('feature_importances')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

#正解率の確認
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

#学習済みモデルの保存
MODEL_SAVE_PATH = 'trained_random_forest_chorus_detector.pkl'
joblib.dump(model, MODEL_SAVE_PATH)
print(f"\n学習済みモデルを '{MODEL_SAVE_PATH}' に保存しました。")

CONFIG_SAVE_PATH = 'feature_extraction_config.json'
import json
config = {
    'SR': SR,
    'N_MFCC': N_MFCC,
    'HOP_LENGTH': HOP_LENGTH
}
with open(CONFIG_SAVE_PATH, 'w') as f:
    json.dump(config, f, indent=4)
print(f"特徴量抽出設定を '{CONFIG_SAVE_PATH}' に保存しました。")