from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logging(log_dir: str | Path) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "app.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(message)s")
    have_file = any(
        isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path
        for h in root.handlers
    )
    if not have_file:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Add a stream handler only if one isn't already present.
    have_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers)
    if not have_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)


def log_event(event: str, payload: dict[str, Any]) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    logging.getLogger("app").info(json.dumps(record, ensure_ascii=False))
