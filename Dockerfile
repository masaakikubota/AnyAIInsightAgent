# Python 3.10 固定
FROM python:3.10-slim

# 環境変数の設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係のインストール
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# アプリケーション本体をコピー
COPY . /app

# Cloud Run は $PORT 環境変数を使用
ENV PORT=8080

# アプリケーションの起動コマンド
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
