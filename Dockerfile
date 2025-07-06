# ベースイメージを、onnxruntimeが要求するCUDA 12.1.1に完全一致させた最強版！
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 環境変数の設定
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# 必要なツールとPythonをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.10 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# python3 -> python になるようにシンボリックリンクを作成
RUN ln -s /usr/bin/python3 /usr/bin/python

# 作業ディレクトリを作成して移動
WORKDIR /app

# 先にrequirements.txtだけコピーして、ライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトのファイルを全部コピー
COPY . .

# サーバーを起動するコマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]