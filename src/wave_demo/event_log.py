from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import threading

from .contracts import LifecycleEvent


class EventLog:
    """Append-only local diagnostics; omit utterance/reply contents by default."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("a", buffering=1)
        self._lock = threading.Lock()

    def __call__(self, event: LifecycleEvent) -> None:
        record = asdict(event)
        record["wall_time"] = datetime.now(timezone.utc).isoformat()
        details = record["details"]
        for name in ("text", "transcript", "response"):
            if name in details:
                details[name + "_characters"] = len(str(details.pop(name)))
        with self._lock:
            self._file.write(json.dumps(record, default=str, allow_nan=False) + "\n")

    def close(self) -> None:
        with self._lock:
            self._file.close()
