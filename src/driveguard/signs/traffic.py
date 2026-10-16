"""Traffic-light colour and stop-sign awareness, built on top of the YOLO detections.

YOLO (COCO) already finds "traffic light" and "stop sign" boxes; this module adds what it cannot:

1. Traffic-light colour: inside each light's box, count bright, saturated pixels that are
   red / yellow / green (HSV) and take the dominant colour. Votes over a few frames remove flicker.
2. "Light ahead": of all lights in view, the largest one near the centre line, in the upper
   part of the frame, is treated as the light for the vehicle's own lane.
3. Stop sign: alert once, per sign, when it is tall enough in the frame to be close.

Limits (be honest in demos): colour is judged from a small crop at low resolution, so lights
that are tiny, blown-out, or backlit may come back UNKNOWN. This is a heuristic, not a trained
classifier, and it has only been checked on synthetic crops.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

from driveguard.config import SignsConfig

if TYPE_CHECKING:
    from driveguard.tracking.tracker import Track


class LightState(Enum):
    UNKNOWN = "unknown"
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


@dataclass(frozen=True)
class SignalReport:
    lights: dict[int, LightState]                 # smoothed colour per traffic-light track
    ahead_id: int | None                          # the light judged to be for our lane
    ahead_state: LightState
    stop_id: int | None                           # a stop sign that is close enough to matter
    events: list[tuple[str, dict]] = field(default_factory=list)  # newly raised (kind, data) events


def classify_light(frame: np.ndarray, xyxy: tuple, cfg: SignsConfig) -> LightState:
    """Dominant lit colour inside one traffic-light box, or UNKNOWN when it is unclear."""
    fh, fw = frame.shape[:2]
    x1, y1 = max(int(xyxy[0]), 0), max(int(xyxy[1]), 0)
    x2, y2 = min(int(xyxy[2]), fw), min(int(xyxy[3]), fh)
    if min(x2 - x1, y2 - y1) < cfg.light_min_size:
        return LightState.UNKNOWN

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lit = (sat >= cfg.light_sat_min) & (val >= cfg.light_val_min)   # OpenCV hue runs 0-179
    counts = {
        LightState.RED: int((lit & ((hue <= 10) | (hue >= 170))).sum()),
        LightState.YELLOW: int((lit & (hue >= 15) & (hue <= 35)).sum()),
        LightState.GREEN: int((lit & (hue >= 40) & (hue <= 95)).sum()),
    }
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_n), (_, second_n) = ranked[0], ranked[1]
    needed = max(2.0, cfg.light_min_fraction * crop.shape[0] * crop.shape[1])
    if best_n < needed or second_n * 2 >= best_n:   # too little colour, or two colours compete
        return LightState.UNKNOWN
    return best


class SignalMonitor:
    def __init__(self, cfg: SignsConfig) -> None:
        self._cfg = cfg
        self._votes: dict[int, deque] = {}
        self._seen: dict[int, int] = {}
        self._stop_announced: set[int] = set()
        self._reported_state = LightState.UNKNOWN
        self._ahead_missing = 0
        self._frame = 0

    def update(self, frame: np.ndarray, tracks: list[Track]) -> SignalReport:
        c = self._cfg
        fh, fw = frame.shape[:2]
        self._frame += 1
        events: list[tuple[str, dict]] = []

        lights: dict[int, LightState] = {}
        ahead: tuple[float, int] | None = None   # (box area, track id) of the best "ahead" candidate
        stop: tuple[float, int] | None = None

        for t in tracks:
            x1, y1, x2, y2 = t.xyxy
            if t.label == "traffic light":
                votes = self._votes.setdefault(t.track_id, deque(maxlen=c.vote_frames))
                votes.append(classify_light(frame, t.xyxy, c))
                self._seen[t.track_id] = self._frame
                lights[t.track_id] = self._smoothed(votes)

                centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
                if abs(centre_x - fw / 2) <= c.ahead_half_width * fw and centre_y <= c.ahead_max_y * fh:
                    area = (x2 - x1) * (y2 - y1)
                    if ahead is None or area > ahead[0]:
                        ahead = (area, t.track_id)
            elif t.label == "stop sign" and (y2 - y1) >= c.stop_min_height * fh:
                self._seen[t.track_id] = self._frame
                if stop is None or (y2 - y1) > stop[0]:
                    stop = (y2 - y1, t.track_id)

        ahead_id = ahead[1] if ahead else None
        ahead_state = lights[ahead_id] if ahead_id is not None else LightState.UNKNOWN

        if ahead_id is None:
            self._ahead_missing += 1
            if self._ahead_missing > c.stale_after:
                self._reported_state = LightState.UNKNOWN   # light left view: announce it again next time
        else:
            self._ahead_missing = 0
            if ahead_state is not LightState.UNKNOWN and ahead_state is not self._reported_state:
                self._reported_state = ahead_state
                events.append(("traffic_light", {"state": ahead_state.value, "track_id": ahead_id}))

        stop_id = stop[1] if stop else None
        if stop_id is not None and stop_id not in self._stop_announced:
            self._stop_announced.add(stop_id)
            events.append(("stop_sign", {"track_id": stop_id}))

        self._prune()
        return SignalReport(lights, ahead_id, ahead_state, stop_id, events)

    def _smoothed(self, votes: deque) -> LightState:
        known = Counter(v for v in votes if v is not LightState.UNKNOWN)
        if not known:
            return LightState.UNKNOWN
        state, n = known.most_common(1)[0]
        return state if n >= self._cfg.min_votes else LightState.UNKNOWN

    def _prune(self) -> None:
        old = [i for i, f in self._seen.items() if self._frame - f > self._cfg.stale_after]
        for i in old:
            del self._seen[i]
            self._votes.pop(i, None)
            self._stop_announced.discard(i)
