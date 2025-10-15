#!/usr/bin/env python3
"""
ローカル実行用スクリプト
GitHubからクローンしたプロジェクトを簡単にローカルで実行できます
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """必要な依存関係をチェック"""
    print("🔍 依存関係をチェック中...")
    
    # Python バージョンチェック
    if sys.version_info < (3, 10):
        print("❌ Python 3.10以上が必要です")
        return False
    
    # requirements.txtの存在チェック
    if not Path("requirements.txt").exists():
        print("❌ requirements.txtが見つかりません")
        return False
    
    print("✅ 依存関係チェック完了")
    return True

def install_dependencies():
    """依存関係をインストール"""
    print("📦 依存関係をインストール中...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ 依存関係のインストール完了")
        return True
    except subprocess.CalledProcessError:
        print("❌ 依存関係のインストールに失敗しました")
        return False

def setup_environment():
    """環境変数を設定"""
    print("🔧 環境変数を設定中...")
    
    # .envファイルの存在チェック
    if not Path(".env").exists():
        print("⚠️  .envファイルが見つかりません")
        print("   Keys.txtから環境変数を作成しますか？ (y/n): ", end="")
        
        if input().lower() == 'y':
            create_env_from_keys()
    
    print("✅ 環境設定完了")

def create_env_from_keys():
    """Keys.txtから.envファイルを作成"""
    try:
        with open("Keys.txt", "r") as f:
            content = f.read().strip()
        
        # Keys.txtの内容を.env形式に変換
        env_content = f"""# 環境変数設定
GEMINI_API_KEY={content.split('\\n')[0] if '\\n' in content else content}
OPENAI_API_KEY={content.split('\\n')[1] if '\\n' in content else ''}
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ .envファイルを作成しました")
    except Exception as e:
        print(f"❌ .envファイルの作成に失敗: {e}")

def run_application():
    """アプリケーションを実行"""
    print("🚀 アプリケーションを起動中...")
    print("📍 アクセス先: http://localhost:25253")
    print("🛑 停止するには Ctrl+C を押してください")
    print("-" * 50)
    
    try:
        # アプリケーションを実行
        subprocess.run([
            sys.executable, "-m", "app.main"
        ], check=True)
    except KeyboardInterrupt:
        print("\\n👋 アプリケーションを停止しました")
    except subprocess.CalledProcessError as e:
        print(f"❌ アプリケーションの実行に失敗: {e}")

def main():
    """メイン処理"""
    print("🎯 AnyAI Marketing Agent - ローカル実行")
    print("=" * 50)
    
    # 依存関係チェック
    if not check_requirements():
        sys.exit(1)
    
    # 依存関係インストール
    if not install_dependencies():
        sys.exit(1)
    
    # 環境設定
    setup_environment()
    
    # アプリケーション実行
    run_application()

if __name__ == "__main__":
    main()
