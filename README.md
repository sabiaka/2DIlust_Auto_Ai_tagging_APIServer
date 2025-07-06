# 2DIlust_Auto_Ai_tagging_APIServer - 画像自動タグ付けAPIサーバー

このプロジェクトは、画像をアップロードすると自動的にタグを付けてくれるFastAPIベースのサーバーです。
[2DIlust_Auto_Ai_tagging_Client](https://github.com/sabiaka/2DIlust_Auto_Ai_tagging_Client)との連携を想定して開発されています。

## 使用モデル

このプロジェクトは [wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/tree/main) モデルを使用して画像の自動タグ付けを行います。このモデルは高精度な画像認識とタグ付けに特化しており、2Dイラストのタグ付けに最適化されています。

## 機能

- 画像の自動タグ付け
- 英語タグから日本語への自動翻訳
- RESTful API エンドポイント
- Docker対応
- GPU対応（CUDA）

## セットアップ方法

### 前提条件

- Python 3.10以上
- Docker（Docker Compose使用時）
- NVIDIA GPU + CUDA（GPU使用時）

### 使い方

1. **リポジトリをクローン**
   ```bash
   git clone <repository-url>
   cd auto_aitag
   ```

1. **必要ファイルのダウンロード**

   [model.safetensors](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.safetensors?download=true) を`models_data`の中に入れる。

2. **Docker Composeで起動**
   ```bash
   docker-compose up --build
   ```

3. **サーバーが起動したら、以下のURLでアクセス可能**
   - API: http://localhost:8000

4. **[2DIlust_Auto_Ai_tagging_Client](https://github.com/sabiaka/2DIlust_Auto_Ai_tagging_Client)から処理を実行**



## API使用方法

### 画像タグ付けエンドポイント

**POST** `/tag`

画像ファイルをアップロードしてタグを取得します。

**リクエスト例:**
```bash
curl -X POST "http://localhost:8000/tag" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

**レスポンス例:**
```json
{
  "tags": ["1人の女の子", "ロングヘア", "カメラ目線", "ウサギの耳"]
}
```

## ファイル構成

```
auto_aitag/
├── main.py              # FastAPIサーバーのメインファイル
├── tagger.py            # 画像タグ付けのロジック
├── Models.py            # 機械学習モデルの定義
├── requirements.txt     # Python依存関係
├── docker-compose.yml   # Docker Compose設定
├── Dockerfile          # Docker設定
└── models_data/        # 学習済みモデルとデータ
    ├── config.json     # モデル設定
    ├── model.safetensors # 学習済みモデル
    ├── model.onnx      # ONNX形式モデル
    ├── top_tags.txt    # タグリスト（英語）
    └── top_tags.csv    # タグ翻訳辞書
```

## 技術仕様

- **フレームワーク**: FastAPI
- **機械学習**: PyTorch
- **画像処理**: Pillow
- **GPU対応**: CUDA（利用可能な場合）
- **モデル**: カスタムVision Transformer