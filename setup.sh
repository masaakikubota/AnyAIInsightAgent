#!/bin/bash
# セットアップスクリプト

set -euo pipefail

cd "$(dirname "$0")"

echo "🎯 AnyAI Marketing Agent - セットアップ"
echo "========================================"

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "❌ python3 コマンドが見つかりません"
    exit 1
fi

# Python バージョンチェック
python_version=$($PYTHON_BIN --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10以上が必要です (現在: $python_version)"
    exit 1
fi

echo "✅ Python バージョン: $python_version"

echo "📚 依存関係をインストール中..."

install_with_python() {
    local python_bin="$1"

    if ! "$python_bin" -m pip --version >/dev/null 2>&1; then
        return 1
    fi

    if ! "$python_bin" -m pip install -r requirements.txt; then
        return 1
    fi

    return 0
}

used_venv=false

if install_with_python "$PYTHON_BIN"; then
    echo "✅ 依存関係をインストールしました ($PYTHON_BIN)"
    echo "   コマンド: $PYTHON_BIN -m uvicorn main:app --host 0.0.0.0 --port 25253"
else
    echo "⚠️  グローバル環境へのインストールに失敗したため、仮想環境(.venv)を使用します"

    if [ ! -d .venv ]; then
        $PYTHON_BIN -m venv .venv
    fi

    VENV_PY="$(pwd)/.venv/bin/python"
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install -r requirements.txt

    echo "✅ 仮想環境 (.venv) に依存関係をインストールしました"
    echo "   コマンド: source .venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 25253"
    used_venv=true
fi

# 実行権限の付与
chmod +x run_local.py

echo "✅ セットアップ完了！"
echo ""
echo "🚀 実行方法:"
if [ "$used_venv" = true ]; then
    echo "   source .venv/bin/activate"
    echo "   uvicorn main:app --host 0.0.0.0 --port 25253"
    echo "   (または) source .venv/bin/activate && python3 run_local.py"
else
    echo "   $PYTHON_BIN -m uvicorn main:app --host 0.0.0.0 --port 25253"
    echo "   (または) $PYTHON_BIN run_local.py"
fi
echo ""
echo "🌐 アクセス先: http://localhost:25253"
