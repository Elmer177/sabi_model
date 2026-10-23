# sabi_model
wavファイルから曲のサビ区間を推定する学習モデル

sabi_model/
├── data/music_dataset/   音源とサビのラベル（変更なし）
├── my_wav_files/         YouTube から取ってきた音源（変更なし）
├── src/
│   ├── paths.py          ★ 共通のパス定義（新規）
│   ├── prep/             データ準備     linktowav.py, urlToWav.ipynb
│   ├── features/         特徴量の抽出   feature_analysis.py (v1), features_v2.py
│   ├── train/            学習           sabi_model.py (v1), train_v2.py
│   ├── evaluate/         評価           evaluate_split.py
│   └── detect/           サビ検出       detection_chorus.py, detection_chorus_multi.py
├── models/               学習済みモデルと特徴量の設定（.pkl, .json）
├── cache/                作り直せる中間ファイル（X.npy, y.npy, features_*.npz/.pkl）
└── results/              評価結果の CSV と図（results/figures/）
