"""Structured event log: one JSON object per line (JSON Lines), plus running counts.

Every alert the system raises (lane departure, critical collision risk, traffic light change,
stop sign) is recorded with its frame number and video time, so a run can be audited afterwards
without watching the video. Plain stdlib only, so it is cheap to import and trivial to test.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


class EventLog:
    """Collects events in memory and, when a path is given, appends them to a .jsonl file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self._fh = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")

    def emit(self, kind: str, frame: int, t: float, **data: Any) -> dict[str, Any]:
        """Record one event. `t` is video time in seconds (frame_index / fps)."""
        event = {"kind": kind, "frame": frame, "t": round(t, 3), **data}
        self.events.append(event)
        self.counts[kind] += 1
        if self._fh:
            self._fh.write(json.dumps(event) + "\n")
            self._fh.flush()   # keep the file useful even if the run is interrupted
        return event

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
