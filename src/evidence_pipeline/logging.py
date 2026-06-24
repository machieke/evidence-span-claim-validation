from __future__ import annotations

import json
import logging as std_logging
from typing import Any, Dict


class JSONFormatter(std_logging.Formatter):
    def format(self, record: std_logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: str = "INFO") -> None:
    handler = std_logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root = std_logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
