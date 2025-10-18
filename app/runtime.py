from __future__ import annotations

import sys


def ensure_runtime_compat() -> None:
    """Fail fast with a clear message when pydantic_core is missing."""
    try:
        import pydantic  # noqa: F401
        import pydantic_core  # type: ignore  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        hint = (
            "pydantic_core が読み込めません。Python 3.13 では未対応のバイナリが原因の可能性があります。\n"
            "対処案:\n"
            "  1) Python 3.12/3.11 で仮想環境を作成し直す (推奨)\n"
            "     pyenv 例: pyenv install 3.12.6 && pyenv local 3.12.6\n"
            "  2) Python 3.13 のまま pydantic/pydantic-core を対応版へ更新 (要Rust等、非推奨)\n"
        )
        raise RuntimeError(hint) from exc


__all__ = ["ensure_runtime_compat"]
