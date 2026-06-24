# 2DIlust_Auto_Ai_tagging_APIServer

画像・動画・PSDファイルを受け取り、自動でタグ付けするFastAPIサーバーです。

現在の標準バックエンドは PixAI Tagger です。旧仕様の `wd-eva02-large-tagger-v3` も `TAGGER_BACKEND=wd` に切り替えることで利用できます。

## 機能

- 画像ファイルの自動タグ付け
- 動画ファイルのシーン分割とタグ集約
- PSDファイル対応
- PixAI Tagger対応
- 英語タグから日本語タグへの変換
- Docker / Docker Compose対応
- NVIDIA GPU対応

## Dockerでセットアップする

### 前提条件

- Docker Desktop または Docker Engine
- Docker Compose
- NVIDIA GPUを使う場合:
  - NVIDIA Driver
  - NVIDIA Container Toolkit
  - `docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi` が成功すること

PixAI Taggerを使う場合、旧WDモデルの `model.safetensors` は不要です。初回起動時に必要なPixAI関連データがダウンロードされるため、ネットワーク接続が必要です。

### 起動手順

1. リポジトリを取得します。

   ```bash
   git clone <repository-url>
   cd 2DIlust_Auto_Ai_tagging_APIServer
   ```

2. Dockerイメージをビルドして起動します。

   ```bash
   docker compose up --build
   ```

3. 起動後、APIにアクセスします。

   - API: http://localhost:8000
   - Web UI: http://localhost:8000
   - Tagger情報: http://localhost:8000/tagger-info

4. 動作確認します。

   ```bash
   curl http://localhost:8000/tagger-info
   ```

   画像タグ付けの例:

   ```bash
   curl -X POST "http://localhost:8000/tag" \
        -H "accept: application/json" \
        -F "file=@your_image.jpg"
   ```

### 停止する

```bash
docker compose down
```

### ログを見る

```bash
docker compose logs -f joytag-api
```

## Docker設定

`docker-compose.yml` の既定値ではPixAI Taggerを使います。

```yaml
environment:
  TAGGER_BACKEND: pixai
  PIXAI_MODEL_NAME: v0.9
  PIXAI_GENERAL_THRESHOLD: "0.30"
  PIXAI_CHARACTER_THRESHOLD: "0.75"
```

しきい値を変えたい場合は、`docker-compose.yml` の値を調整してから再起動してください。

```bash
docker compose up --build
```

## 旧WDモデルを使う場合

旧仕様の `wd-eva02-large-tagger-v3` を使う場合は、次のファイルを `models_data` に配置します。

- `model.safetensors`
- `config.json`
- `selected_tags.csv`

そのうえで `docker-compose.yml` の環境変数を変更します。

```yaml
environment:
  TAGGER_BACKEND: wd
  WD_THRESHOLD: "0.35"
```

`model.onnx` が存在しない場合は、初回起動時に `model.safetensors` からONNXへ変換します。

## API

### `GET /tagger-info`

現在のタグ付けバックエンドとしきい値を返します。

### `GET /`

一枚の画像からStable Diffusion向けのプロンプト候補を作るWeb UIを開きます。

### `POST /analyze`

画像ファイルをアップロードして、英語タグ・日本語表示名・カテゴリ・スコア・プロンプト文字列を返します。

レスポンス例:

```json
{
  "tags": [
    {
      "tag": "long hair",
      "prompt_tag": "long_hair",
      "translated": "ロングヘア",
      "category": "general",
      "score": 0.91
    }
  ],
  "prompt": "long_hair"
}
```

### `POST /tag`

画像または動画ファイルをアップロードして、タグ一覧を返します。

レスポンス例:

```json
{
  "tags": ["1人の女の子", "ロングヘア", "カメラ目線"]
}
```

## 翻訳CSV

タグの日本語化には以下のCSVを使います。

- `models_data/selected_tags.csv`
- `models_data/tag_translations_extra.csv`
- `models_data/pixai_missing_character_translations.csv`
- `models_data/pixai_missing_translations.csv`

`tagger.py` は起動時にこれらを読み込み、PixAIから返った英語タグを日本語タグへ変換します。

## ファイル構成

```text
2DIlust_Auto_Ai_tagging_APIServer/
├── main.py
├── tagger.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── scripts/
└── models_data/
    ├── config.json
    ├── selected_tags.csv
    ├── tag_translations_extra.csv
    ├── pixai_missing_character_translations.csv
    ├── pixai_missing_translations.csv
    ├── model.safetensors
    └── model.onnx
```

`model.safetensors` と `model.onnx` は旧WDモデル利用時のみ必要です。
