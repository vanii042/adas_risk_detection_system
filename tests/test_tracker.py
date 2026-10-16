from driveguard.config import TrackerConfig
from driveguard.detection.types import Detection
from driveguard.tracking.tracker import Tracker

NAMES = {2: "car"}


def _car(x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection((x1, y1, x2, y2), 0.9, 2, "car")


def test_track_id_is_stable_for_moving_object():
    tracker = Tracker(TrackerConfig(), NAMES, frame_rate=30)
    ids = set()
    for i in range(20):
        tracks = tracker.update([_car(100 + 4 * i, 100, 200 + 4 * i, 180)])
        if i >= 8:
            assert len(tracks) == 1
            ids.add(tracks[0].track_id)
    assert len(ids) == 1


def test_two_objects_keep_distinct_stable_ids():
    tracker = Tracker(TrackerConfig(), NAMES, frame_rate=30)
    seen = []
    for i in range(20):
        tracks = tracker.update([
            _car(100 + 4 * i, 100, 200 + 4 * i, 180),
            _car(400 - 4 * i, 120, 500 - 4 * i, 200),
        ])
        if i >= 8:
            assert len(tracks) == 2
            seen.append(frozenset(t.track_id for t in tracks))
    assert len(set(seen)) == 1