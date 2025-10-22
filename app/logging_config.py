from __future__ import annotations

import logging
import os
from logging.config import dictConfig


_CONFIGURED = False


class _AppDebugFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.DEBUG:
            return record.name.startswith("app.")
        return True


def configure_logging() -> None:
    """Configure verbose logging for debugging.

    When ANYAI_LOG_LEVEL is set, that level is used; otherwise default to DEBUG.
    The configuration ensures uvicorn access logs continue to appear while
    application-specific loggers emit detailed context to the terminal.
    """
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger(__name__).debug("Logging already configured; skipping reconfiguration")
        return

    log_level = os.getenv("ANYAI_LOG_LEVEL", "DEBUG").upper()

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "verbose": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "uvicorn": {
                    "format": "%(asctime)s [%(levelname)s] %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "verbose",
                },
                "uvicorn": {
                    "class": "logging.StreamHandler",
                    "formatter": "uvicorn",
                },
            },
            "loggers": {
                "": {  # root logger
                    "handlers": ["console"],
                    "level": log_level,
                },
                "uvicorn.error": {
                    "level": "INFO",
                },
                "uvicorn.access": {
                    "handlers": ["uvicorn"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logging.getLogger().addFilter(_AppDebugFilter())
    logging.getLogger(__name__).debug("Logging configured (level=%s)", log_level)
    _CONFIGURED = True
