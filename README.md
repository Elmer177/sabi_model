# sabi_model
wavファイルから曲のサビ区間を推定する学習モデル

```
sabi_model/
├── data/music_dataset/   音源とサビのラベル学習データ
├── src/
│   ├── paths.py          ★ 共通のパス定義
│   ├── prep/             データ準備     
│   ├── features/         特徴量の抽出   
│   ├── train/            学習          
│   ├── evaluate/         評価           
│   └── detect/           サビ検出      
├── models/               学習済みモデルと特徴量の設定（.pkl, .json）
├── cache/                作り直せる中間ファイル（X.npy, y.npy, features_*.npz/.pkl）
└── results/              評価結果の CSV と図（results/figures/）
```
