"""The per-frame ADAS pipeline: lanes, detection, tracking, collision risk and traffic signs.

`Pipeline.process(frame)` runs every stage on one frame and returns a `FrameResult`;
`Pipeline.annotate(frame, result)` draws that result. Keeping this out of the command-line script means
the whole chain can be driven from tests (or other front-ends) with any detector, for example a scripted
one that needs no YOLO model.

The detector is passed in rather than built here. It only needs a `names` mapping and a
`detect(frame) -> list[Detection]` method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from driveguard.config import Config
from driveguard.detection.types import Detection
from driveguard.events.log import EventLog
from driveguard.lane.departure import DepartureChecker, DepartureState
from driveguard.lane.lane_detector import LaneDetector, LaneResult
from driveguard.risk.collision import CollisionRisk, RiskResult
from driveguard.signs.traffic import SignalMonitor, SignalReport
from driveguard.timing import Stopwatch
from driveguard.tracking.tracker import Track, Tracker
from driveguard.viz.overlay import (
    draw_collision_warning, draw_corridor, draw_debug, draw_departure, draw_hud, draw_lane, draw_signals,
    draw_tracks,
)

STAGES = ("lane", "detect", "track", "risk", "signs")   # names used in FrameResult.timings_ms


class DetectorLike(Protocol):
    names: dict[int, str]

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


@dataclass
class FrameResult:
    index: int
    timestamp: float                              # seconds (index / fps)
    lane: LaneResult
    departure: DepartureState
    detections: list[Detection]
    tracks: list[Track]
    risks: dict[int, RiskResult]
    signals: SignalReport | None                  # None when signs are disabled in the config
    events: list[dict[str, Any]] = field(default_factory=list)   # alerts raised on this frame
    timings_ms: dict[str, float] = field(default_factory=dict)   # per-stage latency


class Pipeline:
    def __init__(self, cfg: Config, detector: DetectorLike, fps: float = 30.0,
                 events: EventLog | None = None) -> None:
        self.cfg = cfg
        self.fps = fps
        self.events = events if events is not None else EventLog()   # in-memory when no file is wanted
        self._detector = detector
        self.lane_detector = LaneDetector(cfg.lane)
        self.departure = DepartureChecker(cfg.departure)
        self.tracker = Tracker(cfg.tracker, detector.names, frame_rate=fps)
        self.collision = CollisionRisk(cfg.risk)
        self.signals = SignalMonitor(cfg.signs) if cfg.signs.enabled else None
        self._n = 0

    def process(self, frame: np.ndarray) -> FrameResult:
        """Run every stage on one (already resized) BGR frame. Does not modify the frame."""
        n, t = self._n, self._n / self.fps
        self._n += 1
        h, w = frame.shape[:2]
        new_events: list[dict[str, Any]] = []
        timings: dict[str, float] = {}
        sw = Stopwatch()

        lane = self.lane_detector.detect(frame)
        state = self.departure.update(lane, w)
        if state.started:
            new_events.append(self.events.emit("lane_departure", n, t, side=state.side,
                                               offset=round(state.offset, 3)))
        timings["lane"] = sw.lap_ms()

        detections = self._detector.detect(frame)
        timings["detect"] = sw.lap_ms()

        tracks = self.tracker.update(detections)
        timings["track"] = sw.lap_ms()

        risks = self.collision.update(tracks, (h, w), t)
        for r in self.collision.new_critical(risks):
            new_events.append(self.events.emit("collision_critical", n, t, track_id=r.track_id,
                                               ttc=round(r.ttc, 2)))
        timings["risk"] = sw.lap_ms()

        report = self.signals.update(frame, tracks) if self.signals else None
        if report:
            for kind, data in report.events:
                new_events.append(self.events.emit(kind, n, t, **data))
        timings["signs"] = sw.lap_ms()

        return FrameResult(n, t, lane, state, detections, tracks, risks, report, new_events, timings)

    def annotate(self, frame: np.ndarray, result: FrameResult, fps: float = 0.0,
                 debug: bool = False) -> np.ndarray:
        """Draw a result onto `frame` in place and return it. `fps` is shown in the HUD."""
        draw_lane(frame, result.lane, warning=result.departure.warning)
        draw_tracks(frame, result.tracks, result.risks)
        draw_collision_warning(frame, result.risks)
        draw_hud(frame, fps, len(result.tracks))
        draw_departure(frame, result.departure)
        if result.signals:
            draw_signals(frame, result.signals, result.tracks)
        if debug and self.lane_detector.debug_edges is not None:
            draw_debug(frame, self.lane_detector.debug_edges, self.lane_detector.roi_polygon(frame.shape[:2]))
            draw_corridor(frame, self.collision.corridor_polygon(frame.shape[:2]))
        return frame
