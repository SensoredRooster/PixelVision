from __future__ import annotations

from datetime import datetime
from pathlib import Path


class EventLogger:
    def __init__(self, log_directory: str | Path):
        self.log_path = Path(log_directory)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.event_log = self.log_path / "events.log"

    def log(self, event: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {event}\n"
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(entry)
