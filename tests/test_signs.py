from types import SimpleNamespace

import cv2
import numpy as np

from driveguard.config import SignsConfig
from driveguard.signs.traffic import LightState, SignalMonitor, classify_light

H, W = 360, 640
RED, YELLOW, GREEN, OFF = (0, 0, 255), (0, 255, 255), (0, 255, 0), (50, 50, 50)   # BGR


def _scene(colour, box=(300, 60, 340, 130)) -> np.ndarray:
    """A dark frame holding one traffic-light housing with a single lamp lit in `colour`."""
    frame = np.zeros((H, W, 3), np.uint8)
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)
    cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), min(x2 - x1, y2 - y1) // 3, colour, -1)
    return frame


def _track(kind, box, tid=1):
    # duck-typed stand-in for tracking.tracker.Track (only these fields are read)
    return SimpleNamespace(track_id=tid, xyxy=box, label=kind)


LIGHT_BOX = (300, 60, 340, 130)


# ------------------------------------------------------------------ colour classification
def test_classifies_each_lit_colour():
    cfg = SignsConfig()
    for colour, expected in ((RED, LightState.RED), (YELLOW, LightState.YELLOW), (GREEN, LightState.GREEN)):
        assert classify_light(_scene(colour), LIGHT_BOX, cfg) is expected


def test_unlit_or_white_lamp_is_unknown():
    cfg = SignsConfig()
    assert classify_light(_scene(OFF), LIGHT_BOX, cfg) is LightState.UNKNOWN
    assert classify_light(_scene((255, 255, 255)), LIGHT_BOX, cfg) is LightState.UNKNOWN   # blown-out lamp


def test_tiny_or_degenerate_boxes_are_unknown_and_do_not_crash():
    cfg = SignsConfig()
    frame = _scene(RED)
    assert classify_light(frame, (300, 60, 304, 64), cfg) is LightState.UNKNOWN     # below light_min_size
    assert classify_light(frame, (30, 30, 30, 30), cfg) is LightState.UNKNOWN       # zero area
    assert classify_light(frame, (-80, -80, 5, 5), cfg) is LightState.UNKNOWN       # mostly off-frame


def test_two_competing_colours_are_unknown():
    frame = _scene(RED)
    cv2.circle(frame, (320, 110), 10, GREEN, -1)   # a second, equally strong lamp
    cv2.circle(frame, (320, 95), 10, RED, -1)
    assert classify_light(frame, LIGHT_BOX, SignsConfig()) is LightState.UNKNOWN


# ------------------------------------------------------------------ traffic-light monitor
def _feed(monitor, colour, frames, box=LIGHT_BOX, tid=1):
    reports = []
    for _ in range(frames):
        reports.append(monitor.update(_scene(colour, box), [_track("traffic light", box, tid)]))
    return reports


def test_colour_is_reported_once_after_enough_votes():
    reports = _feed(SignalMonitor(SignsConfig()), GREEN, 15)
    events = [e for r in reports for e in r.events]
    assert events == [("traffic_light", {"state": "green", "track_id": 1})]
    assert reports[0].ahead_state is LightState.UNKNOWN   # not enough votes yet
    assert reports[-1].ahead_state is LightState.GREEN and reports[-1].ahead_id == 1


def test_a_single_misread_frame_does_not_change_the_state():
    monitor = SignalMonitor(SignsConfig())
    _feed(monitor, GREEN, 10)
    glitch = _feed(monitor, RED, 1)[0]
    assert glitch.ahead_state is LightState.GREEN and glitch.events == []


def test_a_real_change_is_announced():
    monitor = SignalMonitor(SignsConfig())
    _feed(monitor, GREEN, 10)
    reports = _feed(monitor, RED, 12)
    events = [e for r in reports for e in r.events]
    assert events == [("traffic_light", {"state": "red", "track_id": 1})]


def test_light_far_from_the_centre_line_is_not_ahead():
    side_box = (30, 60, 70, 130)
    report = _feed(SignalMonitor(SignsConfig()), RED, 10, box=side_box)[-1]
    assert report.ahead_id is None and report.events == []
    assert report.lights[1] is LightState.RED   # still classified, just not "ours"


def test_light_low_in_the_frame_is_not_ahead():
    low_box = (300, 250, 340, 320)
    assert _feed(SignalMonitor(SignsConfig()), RED, 10, box=low_box)[-1].ahead_id is None


def test_largest_central_light_wins():
    monitor = SignalMonitor(SignsConfig())
    big, small = (300, 40, 350, 140), (420, 60, 436, 90)
    frame = _scene(RED, big)
    cv2.rectangle(frame, small[:2], small[2:], (25, 25, 25), -1)
    cv2.circle(frame, (428, 75), 4, GREEN, -1)
    tracks = [_track("traffic light", small, 2), _track("traffic light", big, 1)]
    for _ in range(10):
        report = monitor.update(frame, tracks)
    assert report.ahead_id == 1 and report.ahead_state is LightState.RED


def test_light_is_announced_again_after_it_left_view():
    cfg = SignsConfig()
    monitor = SignalMonitor(cfg)
    _feed(monitor, GREEN, 10)
    for _ in range(cfg.stale_after + 2):
        monitor.update(np.zeros((H, W, 3), np.uint8), [])
    events = [e for r in _feed(monitor, GREEN, 10, tid=2) for e in r.events]
    assert events == [("traffic_light", {"state": "green", "track_id": 2})]


# ------------------------------------------------------------------ stop sign
def test_distant_stop_sign_is_ignored():
    frame = np.zeros((H, W, 3), np.uint8)
    small = (300, 100, 320, 120)   # 20 px tall = 5.6 % of the frame, below stop_min_height
    report = SignalMonitor(SignsConfig()).update(frame, [_track("stop sign", small)])
    assert report.stop_id is None and report.events == []


def test_close_stop_sign_is_announced_exactly_once():
    monitor = SignalMonitor(SignsConfig())
    frame = np.zeros((H, W, 3), np.uint8)
    close = (280, 80, 340, 160)    # 80 px tall = 22 % of the frame
    reports = [monitor.update(frame, [_track("stop sign", close)]) for _ in range(20)]
    assert sum(len(r.events) for r in reports) == 1
    assert reports[0].events == [("stop_sign", {"track_id": 1})]
    assert all(r.stop_id == 1 for r in reports)   # still flagged while it stays close


def test_two_stop_signs_are_each_announced():
    monitor = SignalMonitor(SignsConfig())
    frame = np.zeros((H, W, 3), np.uint8)
    first = monitor.update(frame, [_track("stop sign", (280, 80, 340, 160), 1)])
    second = monitor.update(frame, [_track("stop sign", (280, 80, 340, 160), 2)])
    assert len(first.events) == 1 and len(second.events) == 1


def test_other_labels_are_ignored():
    frame = np.zeros((H, W, 3), np.uint8)
    report = SignalMonitor(SignsConfig()).update(frame, [_track("car", (100, 100, 300, 250))])
    assert report.lights == {} and report.ahead_id is None and report.stop_id is None
