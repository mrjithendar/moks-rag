"""One-line JSON log records (Loki/Alloy friendly), built with json.dumps so values are escaped."""

from __future__ import annotations

import json
import logging


def emit(
    logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: object
) -> None:
    """Log {"event": event, **fields} as one JSON line."""
    logger.log(level, json.dumps({"event": event, **fields}, default=str))
