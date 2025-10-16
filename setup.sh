#!/bin/bash
# セットアップスクリプト

echo "🎯 AnyAI Marketing Agent - セットアップ"
echo "========================================"

# Python バージョンチェック
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10以上が必要です (現在: $python_version)"
    exit 1
fi

echo "✅ Python バージョン: $python_version"

# 仮想環境の作成
echo "📦 仮想環境を作成中..."
python3 -m venv .venv

# 仮想環境のアクティベート
echo "🔧 仮想環境をアクティベート中..."
source .venv/bin/activate

# 依存関係のインストール
echo "📚 依存関係をインストール中..."
pip install --upgrade pip
pip install -r requirements.txt

# 実行権限の付与
chmod +x run_local.py

echo "✅ セットアップ完了！"
echo ""
echo "🚀 実行方法:"
echo "   python3 run_local.py"
echo ""
echo "🌐 アクセス先: http://localhost:25253"
