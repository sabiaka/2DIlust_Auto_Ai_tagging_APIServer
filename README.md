# 2DIlust_Auto_Ai_tagging_APIServer

画像・動画・PSDファイルを受け取り、自動でタグ付けするFastAPIサーバーです。

現在の標準バックエンドは PixAI Tagger です。旧仕様の `wd-eva02-large-tagger-v3` も `TAGGER_BACKEND=wd` に切り替えることで利用できます。

## このPCのPython構成（PixAI v1.0）

このチェックアウトにはPixAI Tagger v1.0 ONNX FP16対応を追加しています。
親フォルダの `outputs/start-server.cmd` で、専用のPython 3.11環境とRTX 2070を使って起動します。
`outputs/stop-server.cmd` で停止します。APIは `http://127.0.0.1:8000` です。

`PIXAI_MODEL_NAME=v1.0` で新しいバックエンドを選択します。`PIXAI_MODEL_PATH` に
`model_fp16.onnx` のパスを指定し、同じフォルダに `selected_tags.csv` を置きます。
未指定の場合は固定リビジョンのモデルをHugging Faceから取得します。
前処理は1008×1008の黒パディング、CUDAメモリ確保方式は `kSameAsRequested` です。
しきい値は一般0.17、キャラクター0.27、画風0.15、作品0.24、メタ0.17、レーティング0.41です。
各しきい値は `PIXAI_GENERAL_THRESHOLD` などの環境変数で変更できます。
v0.9も `PIXAI_MODEL_NAME=v0.9` で利用でき、その場合の既定しきい値は一般0.30、キャラクター0.75です。

新しいタグのうち既存辞書に訳がないものは英語で表示されます。Dockerの既定設定はv0.9です。

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
- `models_data/tag_translations_ai.csv`（今回の修正版）

今回の修正版は、既存訳の一部を文脈に合わせて修正し、未翻訳タグの一部を追加しています。
すべてのタグの訳を確定したものではありません。作者の識別子は原表記を保持し、
不明な固有名詞や専門語は推測で埋めず、既存辞書または英語表示を使います。
修正版は元の辞書より優先して読み込みます。`tag_translations_extra.csv` の手動設定は最優先です。
Web画面の「CSV翻訳」→「新しい翻訳辞書」から修正版を編集できます。

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
