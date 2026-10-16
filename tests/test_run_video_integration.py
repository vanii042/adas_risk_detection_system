"""End-to-end check of scripts/run_video.py with a scripted detector.

The real tracker, lane detector, risk, signs and event log all run; only YOLO is replaced, so the test needs
no model download and exercises the wiring that unit tests cannot (events reaching the log, summary printing).
"""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from driveguard.detection.types import Detection

ROOT = Path(__file__).resolve().parents[1]
H, W, FPS, N = 360, 640, 30, 40
LIGHT_BOX = (300, 60, 340, 130)
STOP_BOX = (420, 80, 480, 160)    # 80 px tall: close enough to trigger the stop-sign alert


def _load_run_video():
    spec = importlib.util.spec_from_file_location("run_video", ROOT / "scripts" / "run_video.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_clip(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(N):
        frame = np.full((H, W, 3), 60, np.uint8)
        x1, y1, x2, y2 = LIGHT_BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 13, (0, 255, 0), -1)   # green lamp
        writer.write(frame)
    writer.release()


class _ScriptedDetector:
    """Stands in for detection.detector.Detector: always sees one traffic light and one stop sign."""

    names = {9: "traffic light", 11: "stop sign"}

    def __init__(self, cfg):
        pass

    def warmup(self):
        pass

    def detect(self, frame):
        return [Detection(tuple(map(float, LIGHT_BOX)), 0.9, 9, "traffic light"),
                Detection(tuple(map(float, STOP_BOX)), 0.9, 11, "stop sign")]


def test_run_video_logs_sign_events_and_prints_summary(tmp_path, monkeypatch, capsys):
    clip, events_path, out = tmp_path / "clip.mp4", tmp_path / "events.jsonl", tmp_path / "out.mp4"
    _write_clip(clip)

    run_video = _load_run_video()
    monkeypatch.setattr(run_video, "Detector", _ScriptedDetector)
    monkeypatch.setattr(sys, "argv", [
        "run_video.py", "--source", str(clip), "--output", str(out), "--events", str(events_path),
        "--config", str(ROOT / "configs" / "default.yaml"),
    ])
    run_video.main()

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    kinds = [e["kind"] for e in events]
    assert kinds.count("stop_sign") == 1                    # announced once, not every frame
    lights = [e for e in events if e["kind"] == "traffic_light"]
    assert len(lights) == 1 and lights[0]["state"] == "green"
    assert all(set(e) >= {"kind", "frame", "t"} for e in events)

    summary = capsys.readouterr().out
    assert "Frames processed : 40" in summary
    assert "traffic_light 1" in summary and "stop_sign 1" in summary
    assert out.exists() and out.stat().st_size > 0


def test_signs_can_be_disabled_from_the_config(tmp_path, monkeypatch, capsys):
    clip, events_path = tmp_path / "clip.mp4", tmp_path / "events.jsonl"
    _write_clip(clip)
    cfg = tmp_path / "no_signs.yaml"
    cfg.write_text("signs:\n  enabled: false\n", encoding="utf-8")

    run_video = _load_run_video()
    monkeypatch.setattr(run_video, "Detector", _ScriptedDetector)
    monkeypatch.setattr(sys, "argv", [
        "run_video.py", "--source", str(clip), "--output", str(tmp_path / "out.mp4"),
        "--events", str(events_path), "--config", str(cfg), "--max-frames", "10",
    ])
    run_video.main()

    assert events_path.read_text(encoding="utf-8") == ""
    assert "Events logged    : none" in capsys.readouterr().out
