"""Multi-object tracking: links per-frame detections into persistent track IDs."""
from dataclasses import dataclass

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from driveguard.config import TrackerConfig
from driveguard.detection.types import Detection


@dataclass(frozen=True)
class Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str


class Tracker:
    """ByteTrack behind our own interface so the algorithm can be swapped later."""

    def __init__(self, cfg: TrackerConfig, names: dict[int, str], frame_rate: float = 30.0) -> None:
        self._names = names
        self._tracker = ByteTrackTracker(
            frame_rate=frame_rate, lost_track_buffer=cfg.lost_track_buffer
        )

    def update(self, detections: list[Detection]) -> list[Track]:
        sv_dets = sv.Detections(
            xyxy=np.array([d.xyxy for d in detections], dtype=np.float32).reshape(-1, 4),
            confidence=np.array([d.confidence for d in detections], dtype=np.float32),
            class_id=np.array([d.class_id for d in detections], dtype=int),
        )
        tracked = self._tracker.update(sv_dets)
        if tracked.tracker_id is None:
            return []

        n = len(tracked)
        conf = tracked.confidence if tracked.confidence is not None else np.zeros(n)
        cls = tracked.class_id if tracked.class_id is not None else np.full(n, -1)
        return [
            Track(
                int(tid),
                tuple(float(v) for v in box),
                float(c),
                int(k),
                self._names.get(int(k), str(k)),
            )
            for box, c, k, tid in zip(tracked.xyxy, conf, cls, tracked.tracker_id)
            if tid >= 0
        ]