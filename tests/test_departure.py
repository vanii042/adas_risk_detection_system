from driveguard.config import DepartureConfig
from driveguard.lane.departure import DepartureChecker
from driveguard.lane.lane_detector import LaneLine, LaneResult

W = 640


def _lane(left_x: float | None, right_x: float | None) -> LaneResult:
    left = LaneLine(left_x, left_x + 40) if left_x is not None else None
    right = LaneLine(right_x, right_x - 40) if right_x is not None else None
    return LaneResult(left, right, 260, 200, left is not None, right is not None)


def _drive(checker: DepartureChecker, lane: LaneResult, frames: int):
    return [checker.update(lane, W) for _ in range(frames)]


def test_centred_vehicle_has_no_warning():
    states = _drive(DepartureChecker(DepartureConfig()), _lane(160, 480), 30)
    assert not any(s.warning for s in states)
    assert abs(states[-1].offset) < 0.01


def test_drift_triggers_one_warning_with_side():
    checker = DepartureChecker(DepartureConfig())
    _drive(checker, _lane(160, 480), 10)
    # lane slides left of the vehicle => vehicle drifts right (offset ~ +0.4)
    states = _drive(checker, _lane(32, 352), 12)
    assert states[-1].warning and states[-1].side == "right"
    assert sum(s.started for s in states) == 1
    assert not states[0].warning  # needs trigger_frames consecutive frames first


def test_warning_has_hysteresis_and_then_clears():
    cfg = DepartureConfig()
    checker = DepartureChecker(cfg)
    _drive(checker, _lane(160, 480), 5)
    _drive(checker, _lane(32, 352), 8)           # offset ~ +0.4 -> warning
    held = _drive(checker, _lane(88, 408), 20)   # offset ~ +0.25: between clear and warn
    assert all(s.warning for s in held)
    cleared = _drive(checker, _lane(160, 480), cfg.clear_frames + 1)
    assert not cleared[-1].warning


def test_single_visible_line_uses_remembered_lane_width():
    checker = DepartureChecker(DepartureConfig())
    _drive(checker, _lane(160, 480), 5)          # learns a 320 px lane
    state = _drive(checker, _lane(160, None), 1)[0]
    assert state.offset is not None and abs(state.offset) < 0.01


def test_single_line_without_history_gives_no_offset():
    state = DepartureChecker(DepartureConfig()).update(_lane(160, None), W)
    assert state.offset is None and not state.warning


def test_implausible_lane_width_is_ignored():
    state = DepartureChecker(DepartureConfig()).update(_lane(300, 330), W)  # 30 px wide
    assert state.offset is None


def test_warning_clears_when_lane_is_lost_for_long():
    cfg = DepartureConfig()
    checker = DepartureChecker(cfg)
    _drive(checker, _lane(32, 352), 10)
    assert checker.update(_lane(32, 352), W).warning
    states = _drive(checker, _lane(None, None), cfg.clear_frames * 2 + 2)
    assert not states[-1].warning