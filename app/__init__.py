"""Top-level package exports with lazy loading.

FastAPI and other optional dependencies are imported only when the
corresponding attribute is requested. This keeps lightweight imports like
``app.models`` usable in environments where the web stack is not installed,
such as unit tests that only exercise pure-Python helpers.
"""

from importlib import import_module
from typing import Any

__all__ = ["app", "create_app", "run"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    module = import_module(".main", __name__)
    return getattr(module, name)
