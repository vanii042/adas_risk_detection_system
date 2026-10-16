"""Drive the whole per-frame chain (lanes, tracking, risk, signs, events, drawing) with a scripted detector.

No YOLO model is loaded, so these run in a second or two and can exercise scenarios that a real model
cannot be made to produce on demand.
"""
from pathlib import Path

import cv2
import numpy as np

from driveguard.config import SignsConfig, load_config
from driveguard.detection.types import Detection
from driveguard.events.log import EventLog
from driveguard.pipeline import STAGES, Pipeline
from driveguard.risk.collision import RiskLevel

ROOT = Path(__file__).resolve().parents[1]
H, W, FPS = 360, 640, 30
NAMES = {2: "car", 9: "traffic light", 11: "stop sign"}


class Scripted:
    """Detector stand-in: `plan(frame_index)` returns that frame's detections."""

    names = NAMES

    def __init__(self, plan):
        self._plan, self._i = plan, 0

    def detect(self, frame):
        dets = self._plan(self._i)
        self._i += 1
        return dets


def _car(i, growth=1.12, w0=40):
    """A car straight ahead whose box grows by `growth` per frame, feet at 65 % of the frame height."""
    w = w0 * growth ** i
    foot = 0.65 * H
    return Detection((W / 2 - w / 2, foot - 0.8 * w, W / 2 + w / 2, foot), 0.9, 2, "car")


def _blank():
    return np.full((H, W, 3), 60, np.uint8)


def _pipeline(plan, **kwargs):
    return Pipeline(load_config(ROOT / "configs" / "default.yaml"), Scripted(plan), fps=FPS, **kwargs)


def test_result_carries_every_stage_and_timing():
    pipe = _pipeline(lambda i: [])
    result = pipe.process(_blank())
    assert result.index == 0 and result.timestamp == 0.0
    assert result.tracks == [] and result.risks == {} and result.events == []
    assert set(result.timings_ms) == set(STAGES) and all(v >= 0 for v in result.timings_ms.values())
    assert result.signals is not None   # signs are enabled by default


def test_frame_index_and_timestamp_advance_with_fps():
    pipe = _pipeline(lambda i: [])
    results = [pipe.process(_blank()) for _ in range(5)]
    assert [r.index for r in results] == [0, 1, 2, 3, 4]
    assert results[3].timestamp == 3 / FPS


def test_process_does_not_modify_the_frame():
    pipe = _pipeline(lambda i: [_car(i)])
    frame = _blank()
    pipe.process(frame)
    assert (frame == _blank()).all()


def test_approaching_car_raises_one_critical_event_with_ttc():
    log = EventLog()
    pipe = _pipeline(lambda i: [_car(i)], events=log)
    results = [pipe.process(_blank()) for _ in range(25)]

    critical = [e for r in results for e in r.events if e["kind"] == "collision_critical"]
    assert len(critical) == 1                                   # once per object, not every frame
    assert 0 < critical[0]["ttc"] <= 1.5
    assert log.counts["collision_critical"] == 1                # also reached the shared event log
    assert max(x.level for r in results for x in r.risks.values()) == RiskLevel.CRITICAL


def test_car_that_keeps_its_size_never_alerts():
    pipe = _pipeline(lambda i: [_car(0)])
    results = [pipe.process(_blank()) for _ in range(30)]
    assert not any(r.events for r in results)
    assert all(not r.risks or max(x.level for x in r.risks.values()) == RiskLevel.SAFE for r in results)


def test_stop_sign_alert_is_logged_once():
    stop = Detection((420.0, 80.0, 480.0, 160.0), 0.9, 11, "stop sign")
    pipe = _pipeline(lambda i: [stop])
    events = [e for _ in range(20) for e in pipe.process(_blank()).events]
    assert [e["kind"] for e in events] == ["stop_sign"]
    assert events[0]["frame"] >= 0 and "t" in events[0]


def test_disabling_signs_in_the_config_skips_the_stage():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.signs = SignsConfig(enabled=False)
    stop = Detection((420.0, 80.0, 480.0, 160.0), 0.9, 11, "stop sign")
    pipe = Pipeline(cfg, Scripted(lambda i: [stop]), fps=FPS)
    results = [pipe.process(_blank()) for _ in range(10)]
    assert all(r.signals is None and r.events == [] for r in results)


def test_lane_departure_event_when_the_lane_drifts():
    def road(shift):
        f = np.full((H, W, 3), 95, np.uint8)
        cv2.line(f, (60 - shift, H), (290 - shift, int(.42 * H)), (255, 255, 255), 5)
        cv2.line(f, (580 - shift, H), (350 - shift, int(.42 * H)), (255, 255, 255), 5)
        return f

    pipe = _pipeline(lambda i: [])
    events = []
    for i in range(110):
        events += pipe.process(road(0 if i < 30 else min((i - 30) * 3, 110))).events
    departures = [e for e in events if e["kind"] == "lane_departure"]
    assert len(departures) == 1 and departures[0]["side"] == "right"


def test_annotate_draws_in_place_and_returns_the_same_frame():
    pipe = _pipeline(lambda i: [_car(i)])
    frame = _blank()
    result = None
    for i in range(12):
        frame = _blank()
        result = pipe.process(frame)
    out = pipe.annotate(frame, result, fps=12.0)
    assert out is frame and (frame != _blank()).any()


def test_debug_overlay_adds_the_roi_and_corridor():
    pipe = _pipeline(lambda i: [])
    plain, debug = _blank(), _blank()
    result = pipe.process(plain)
    pipe.annotate(plain, result)
    pipe.annotate(debug, result, debug=True)
    assert (debug != plain).any()
