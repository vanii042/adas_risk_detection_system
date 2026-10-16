"""Smoke tests: every overlay draws without crashing and actually changes pixels."""
from types import SimpleNamespace

import numpy as np

from driveguard.lane.departure import DepartureState
from driveguard.lane.lane_detector import LaneLine, LaneResult
from driveguard.risk.collision import RiskLevel, RiskResult
from driveguard.signs.traffic import LightState, SignalReport
from driveguard.viz import overlay

H, W = 360, 640


def _blank():
    return np.zeros((H, W, 3), np.uint8)


def _changed(fn, *args):
    frame = _blank()
    fn(frame, *args)
    return bool(frame.any())


def test_signals_overlay_draws_light_badge_and_stop_banner():
    track = SimpleNamespace(track_id=1, xyxy=(300, 60, 340, 130))
    report = SignalReport({1: LightState.RED}, 1, LightState.RED, 5)
    assert _changed(overlay.draw_signals, report, [track])


def test_signals_overlay_is_silent_when_nothing_is_in_view():
    report = SignalReport({}, None, LightState.UNKNOWN, None)
    assert not _changed(overlay.draw_signals, report, [])


def test_unknown_light_state_still_draws_a_badge():
    track = SimpleNamespace(track_id=1, xyxy=(300, 60, 340, 130))
    assert _changed(overlay.draw_signals, SignalReport({1: LightState.UNKNOWN}, 1, LightState.UNKNOWN, None), [track])


def test_collision_banner_only_from_warning_up():
    caution = {1: RiskResult(1, RiskLevel.CAUTION, True, 4.0)}
    critical = {1: RiskResult(1, RiskLevel.CRITICAL, True, 1.0)}
    assert not _changed(overlay.draw_collision_warning, caution)
    assert _changed(overlay.draw_collision_warning, critical)
    assert not _changed(overlay.draw_collision_warning, {})


def test_departure_banner_and_lane_draw():
    assert _changed(overlay.draw_departure, DepartureState(0.4, "right", True, True))
    lane = LaneResult(LaneLine(160, 200), LaneLine(480, 440), 260, 200, True, True)
    assert _changed(overlay.draw_lane, lane)
