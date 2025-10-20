import asyncio
import inspect
from typing import Any

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    func: Any = pyfuncitem.obj
    if inspect.iscoroutinefunction(func):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(func(**pyfuncitem.funcargs))
        finally:
            loop.close()
        return True
    return None
