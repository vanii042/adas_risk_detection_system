import json

from driveguard.events.log import EventLog


def test_events_are_kept_in_memory_and_counted():
    log = EventLog()
    log.emit("lane_departure", 10, 0.3333, side="left", offset=-0.41)
    log.emit("lane_departure", 90, 3.0, side="right", offset=0.4)
    log.emit("stop_sign", 120, 4.0, track_id=7)
    assert log.counts == {"lane_departure": 2, "stop_sign": 1}
    assert log.events[0] == {"kind": "lane_departure", "frame": 10, "t": 0.333, "side": "left", "offset": -0.41}


def test_events_are_written_as_json_lines(tmp_path):
    path = tmp_path / "nested" / "events.jsonl"   # parent folder is created on demand
    with EventLog(path) as log:
        log.emit("collision_critical", 42, 1.4, track_id=3, ttc=1.2)
        log.emit("traffic_light", 50, 1.67, state="red", track_id=9)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["collision_critical", "traffic_light"]
    assert json.loads(lines[0])["ttc"] == 1.2


def test_file_is_usable_before_close(tmp_path):
    """Each event is flushed immediately, so an interrupted run still leaves a readable log."""
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.emit("stop_sign", 1, 0.03, track_id=1)
    assert json.loads(path.read_text(encoding="utf-8"))["kind"] == "stop_sign"
    log.close()
    log.close()   # closing twice is harmless
