"""Collision risk: ego-corridor membership + a time-to-collision (TTC) proxy from bbox growth.

1. Ego corridor: a trapezoid in front of the car. An object is "in path" when its
   bottom-centre point (where it touches the road) lies inside it.
2. TTC proxy: an approaching object's apparent width w scales as 1 / distance, so
   d(ln w)/dt = 1 / TTC. TTC is 1 / (robust Theil-Sen slope of ln w over the last few frames).
   No camera calibration is needed, but it is a proxy, not metric time-to-collision.
3. Risk level comes from corridor membership and TTC thresholds.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from statistics import median
from typing import TYPE_CHECKING

from driveguard.config import RiskConfig

if TYPE_CHECKING:  # keeps this module importable without the tracking dependencies
    from driveguard.tracking.tracker import Track


class RiskLevel(IntEnum):
    SAFE = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3


@dataclass(frozen=True)
class RiskResult:
    track_id: int
    level: RiskLevel
    in_corridor: bool
    ttc: float | None   # seconds; None when not approaching or not enough history


@dataclass
class _History:
    t: deque = field(default_factory=deque)
    log_w: deque = field(default_factory=deque)
    raw_levels: deque = field(default_factory=deque)   # most recent un-debounced levels
    last_seen: int = 0


class CollisionRisk:
    def __init__(self, cfg: RiskConfig) -> None:
        self._cfg = cfg
        self._hist: dict[int, _History] = {}
        self._frame = 0
        self._critical_ids: set[int] = set()

    def corridor_polygon(self, frame_shape: tuple[int, int]) -> list[tuple[int, int]]:
        """Corner points (x, y) in pixels, for drawing. frame_shape = (height, width)."""
        c, (h, w) = self._cfg, frame_shape
        cx = w / 2
        ty, by = c.corridor_top_y * h, c.corridor_bottom_y * h
        th, bh = c.corridor_top_half_width * w, c.corridor_bottom_half_width * w
        return [(round(cx - bh), round(by)), (round(cx - th), round(ty)),
                (round(cx + th), round(ty)), (round(cx + bh), round(by))]

    def update(self, tracks: list[Track], frame_shape: tuple[int, int],
               timestamp: float) -> dict[int, RiskResult]:
        """frame_shape = (height, width); timestamp in seconds (frame_index / fps)."""
        h, w = frame_shape
        self._frame += 1
        out: dict[int, RiskResult] = {}

        for tr in tracks:
            if tr.label not in self._cfg.classes:
                continue
            x1, _, x2, y2 = tr.xyxy
            hist = self._hist.setdefault(tr.track_id, _History())
            hist.t.append(timestamp)
            hist.log_w.append(math.log(max(x2 - x1, 1.0)))
            while len(hist.t) > self._cfg.window:
                hist.t.popleft()
                hist.log_w.popleft()
            hist.last_seen = self._frame

            in_corridor = self._in_corridor((x1 + x2) / 2, y2, h, w)
            ttc = self._ttc(hist)
            level = self._confirmed(hist, self._level(in_corridor, ttc))
            out[tr.track_id] = RiskResult(tr.track_id, level, in_corridor, ttc)

        for tid in [i for i, hs in self._hist.items() if self._frame - hs.last_seen > self._cfg.stale_after]:
            del self._hist[tid]
        return out

    def _in_corridor(self, x: float, y: float, h: int, w: int) -> bool:
        c = self._cfg
        ty, by = c.corridor_top_y * h, c.corridor_bottom_y * h
        if y < ty:
            return False
        frac = min((y - ty) / max(by - ty, 1e-6), 1.0)   # 0 at the top edge, 1 at/below the bottom edge
        half = c.corridor_top_half_width + frac * (c.corridor_bottom_half_width - c.corridor_top_half_width)
        return abs(x - w / 2) <= half * w

    def _ttc(self, hist: _History) -> float | None:
        n = len(hist.t)
        if n < self._cfg.min_history:
            return None
        ts, ys = list(hist.t), list(hist.log_w)
        # Theil-Sen: the median of all pairwise slopes. Identical to least squares on clean data,
        # but one glitchy bounding box cannot drag the estimate (a least-squares fit would let a
        # single wide box fake an approach for the whole window and trigger a false alert).
        slopes = [(ys[j] - ys[i]) / (ts[j] - ts[i])
                  for i in range(n) for j in range(i + 1, n) if ts[j] - ts[i] > 1e-9]
        if not slopes:
            return None
        slope = median(slopes)
        return None if slope < self._cfg.min_slope else 1.0 / slope

    def _confirmed(self, hist: _History, raw: RiskLevel) -> RiskLevel:
        """Debounce: report the highest level that held for the last `confirm_frames` frames.

        Bounding-box width is noisy, so a single jittery frame must not raise a warning.
        """
        k = max(self._cfg.confirm_frames, 1)
        hist.raw_levels.append(raw)
        while len(hist.raw_levels) > k:
            hist.raw_levels.popleft()
        if len(hist.raw_levels) < k:
            return RiskLevel.SAFE
        return min(hist.raw_levels)

    def _level(self, in_corridor: bool, ttc: float | None) -> RiskLevel:
        c = self._cfg
        if not in_corridor or ttc is None:
            return RiskLevel.SAFE
        if ttc <= c.ttc_critical:
            return RiskLevel.CRITICAL
        if ttc <= c.ttc_warning:
            return RiskLevel.WARNING
        if ttc <= c.ttc_caution:
            return RiskLevel.CAUTION
        return RiskLevel.SAFE

    def new_critical(self, results: dict[int, RiskResult]) -> list[RiskResult]:
        """Results that just reached CRITICAL (once per object), for event logging in Milestone 5."""
        fresh = [r for r in results.values()
                 if r.level == RiskLevel.CRITICAL and r.track_id not in self._critical_ids]
        self._critical_ids.update(r.track_id for r in fresh)
        self._critical_ids &= set(self._hist)
        return fresh
