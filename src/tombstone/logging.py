"""Structured JSON logs to **stderr**, never stdout (stdout is the MCP stdio transport).

Log records carry artifact ids, store names, counts and measurements. They never carry content,
raw subject ids, canaries or embeddings (Hard Rule 7). ``redact()`` is applied to every message.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

_LOGGER_NAME = "tombstone"
_configured = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "tombstone_extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info and record.exc_info[1] is not None:
            payload["exc"] = type(record.exc_info[1]).__name__
            payload["exc_msg"] = str(record.exc_info[1])
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def configure(level: str | None = None) -> logging.Logger:
    """Idempotent. Level from ``TOMBSTONE_LOG_LEVEL`` (default ``info``)."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        _configured = True
    lvl = (level or os.environ.get("TOMBSTONE_LOG_LEVEL", "info")).upper()
    logger.setLevel(getattr(logging, lvl, logging.INFO))
    return logger


def get_logger(name: str = "") -> logging.Logger:
    configure()
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


def log(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    """Log with structured fields. Field values must be ids, names, numbers — never content."""
    logger.log(level, msg, extra={"tombstone_extra": fields})
