from types import SimpleNamespace

from driveguard.config import RiskConfig, load_config
from driveguard.risk.collision import CollisionRisk, RiskLevel

H, W, FPS = 360, 640, 30


def _track(box, label="car", tid=1):
    # duck-typed stand-in for tracking.tracker.Track (only these three fields are read)
    return SimpleNamespace(track_id=tid, xyxy=box, label=label)


def _approach(cx, growth=1.15, n=12, w0=40, label="car"):
    """Boxes of an object whose width grows by `growth` per frame, feet at y = 0.65 * H."""
    boxes, w, foot = [], w0, 0.65 * H
    for _ in range(n):
        boxes.append((cx - w / 2, foot - w * 0.8, cx + w / 2, foot))
        w *= growth
    return boxes


def _run(boxes, label="car", cfg=None):
    risk = CollisionRisk(cfg or RiskConfig())
    result = None
    for i, b in enumerate(boxes):
        result = risk.update([_track(b, label)], (H, W), i / FPS).get(1)
    return result


def test_approaching_object_in_path_is_high_risk():
    r = _run(_approach(W / 2))
    assert r.in_corridor and r.ttc is not None
    assert r.level >= RiskLevel.WARNING


def test_same_object_outside_corridor_is_safe():
    r = _run(_approach(40))
    assert not r.in_corridor and r.level == RiskLevel.SAFE


def test_constant_size_is_safe():
    r = _run(_approach(W / 2, growth=1.0))
    assert r.ttc is None and r.level == RiskLevel.SAFE


def test_receding_object_is_safe():
    r = _run(_approach(W / 2, growth=0.95))
    assert r.ttc is None and r.level == RiskLevel.SAFE


def test_slower_approach_gives_lower_risk_than_fast_approach():
    fast = _run(_approach(W / 2, growth=1.15))
    slow = _run(_approach(W / 2, growth=1.04))
    assert slow.ttc > fast.ttc
    assert slow.level <= fast.level


def test_non_vehicle_classes_are_ignored():
    risk = CollisionRisk(RiskConfig())
    out = risk.update([_track((10, 10, 50, 50), "traffic light")], (H, W), 0.0)
    assert out == {}


def test_new_critical_fires_once_per_object():
    risk = CollisionRisk(RiskConfig())
    fired = 0
    for i, b in enumerate(_approach(W / 2, growth=1.25, n=14)):
        res = risk.update([_track(b)], (H, W), i / FPS)
        fired += len(risk.new_critical(res))
    assert fired == 1


def test_corridor_polygon_is_inside_frame_and_symmetric():
    poly = CollisionRisk(RiskConfig()).corridor_polygon((H, W))
    assert len(poly) == 4
    assert poly[0][0] + poly[3][0] == W and poly[1][0] + poly[2][0] == W


def test_every_config_has_a_sane_corridor():
    from pathlib import Path
    for path in (Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"):
        r = load_config(path).risk
        assert 0 < r.corridor_top_y < r.corridor_bottom_y <= 1
        assert r.ttc_critical < r.ttc_warning < r.ttc_caution


def _first_frame_at_or_above(level, confirm_frames):
    cfg = RiskConfig(confirm_frames=confirm_frames)
    risk = CollisionRisk(cfg)
    for i, b in enumerate(_approach(W / 2, growth=1.2, n=20)):
        res = risk.update([_track(b)], (H, W), i / FPS)
        if res[1].level >= level:
            return i
    return None


def test_debounce_delays_a_warning_by_confirm_frames():
    immediate = _first_frame_at_or_above(RiskLevel.WARNING, confirm_frames=1)
    debounced = _first_frame_at_or_above(RiskLevel.WARNING, confirm_frames=4)
    assert immediate is not None and debounced is not None
    assert debounced >= immediate + 3


def test_a_one_frame_spike_never_raises_a_warning():
    risk = CollisionRisk(RiskConfig(confirm_frames=3))
    steady = [(300, 200, 340, 240)] * 10                  # constant size: no approach
    spike = [(280, 180, 360, 240)]                        # one frame with a much wider box
    levels = [risk.update([_track(b)], (H, W), i / FPS)[1].level
              for i, b in enumerate(steady + spike + steady)]
    assert max(levels) == RiskLevel.SAFE
