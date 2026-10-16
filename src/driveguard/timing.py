"""Tiny stopwatch for per-stage latency measurement."""
import time


class Stopwatch:
    def __init__(self) -> None:
        self._t = time.perf_counter()

    def lap_ms(self) -> float:
        now = time.perf_counter()
        elapsed = (now - self._t) * 1000.0
        self._t = now
        return elapsed