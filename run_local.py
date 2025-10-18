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

def install_dependencies() -> str | None:
    """依存関係をインストールし、利用するPython実行パスを返す."""
    print("📦 依存関係をインストール中...")

    pip_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        "requirements.txt",
    ]

    result = subprocess.run(
        pip_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)

    if result.returncode == 0:
        print("✅ 依存関係のインストール完了")
        return sys.executable

    if "externally-managed-environment" in result.stderr:
        return _install_with_virtualenv()

    print("❌ 依存関係のインストールに失敗しました")
    return None


def _install_with_virtualenv() -> str | None:
    """PEP 668環境向けに仮想環境を作成して依存関係をインストールする."""
    print("⚠️  システム管理下のPython環境が検出されました。仮想環境(.venv)を作成します。")

    venv_dir = Path(".venv")
    if not venv_dir.exists():
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"❌ 仮想環境の作成に失敗しました: {exc}")
            return None

    venv_python = _resolve_venv_python(venv_dir)

    try:
        subprocess.run(
            [
                venv_python,
                "-m",
                "pip",
                "install",
                "-r",
                "requirements.txt",
            ],
            check=True,
        )
        print("✅ 仮想環境で依存関係をインストールしました (.venv)")
        print("   次回以降は .venv/bin/activate (またはScripts\\activate) を利用すると便利です。")
        return venv_python
    except subprocess.CalledProcessError as exc:
        print(f"❌ 仮想環境への依存関係インストールに失敗しました: {exc}")
        return None


def _resolve_venv_python(venv_dir: Path) -> str:
    """仮想環境内のPython実行ファイルパスを取得する."""
    if os.name == "nt":
        python_path = venv_dir / "Scripts" / "python.exe"
    else:
        python_path = venv_dir / "bin" / "python"
    return str(python_path)

def setup_environment() -> dict[str, str]:
    """環境変数を設定し、上書きする値を返す."""
    print("🔧 環境変数を設定中...")

    print("   APIキーを入力してください（未設定の場合は空のままEnter）")
    values = _prompt_for_keys()

    env_overrides: dict[str, str] = {}
    for key, value in values.items():
        if value:
            env_overrides[key] = value

    if not env_overrides:
        print("⚠️  APIキーが入力されませんでした。既存の環境変数を利用します。")
    else:
        for key in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
            if key in env_overrides:
                print(f"✅ {key} を設定しました")

    print("✅ 環境設定完了")
    return env_overrides


def _prompt_for_keys() -> dict[str, str]:
    """対話的にAPIキーを入力させる."""
    gemini_key = input("   GEMINI_API_KEY (未設定の場合は空のままEnter): ").strip()
    openai_key = input("   OPENAI_API_KEY (未設定の場合は空のままEnter): ").strip()
    return {"GEMINI_API_KEY": gemini_key, "OPENAI_API_KEY": openai_key}


def run_application(python_exec: str, env_overrides: dict[str, str]):
    """アプリケーションを実行"""
    print("🚀 アプリケーションを起動中...")
    print("📍 アクセス先: http://localhost:25259")
    print("🛑 停止するには Ctrl+C を押してください")
    print("-" * 50)

    try:
        # 環境変数でポートを指定してアプリケーションを実行
        env = os.environ.copy()
        env['PORT'] = '25259'
        env.update(env_overrides)
        subprocess.run([
            python_exec, "-m", "app.main"
        ], check=True, env=env)
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
    python_exec = install_dependencies()
    if not python_exec:
        sys.exit(1)
    
    # 環境設定
    env_overrides = setup_environment()

    # アプリケーション実行
    run_application(python_exec, env_overrides)

if __name__ == "__main__":
    main()
