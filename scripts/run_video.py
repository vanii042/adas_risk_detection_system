"""Run lane detection + departure warning, object detection, tracking, collision risk and
traffic light / stop sign awareness on a video. Alerts are also written to a JSON Lines event log.

The per-frame work lives in driveguard.pipeline.Pipeline; this script only handles the command line,
video input/output, the live preview and the end-of-run summary.

Usage (from project root, venv active):
    python scripts\\run_video.py --source data\\samples\\clip.mp4 --config configs\\highway.yaml
"""
import argparse
import time

import cv2
import numpy as np
import psutil

from driveguard.config import load_config
from driveguard.detection.detector import Detector
from driveguard.events.log import EventLog
from driveguard.pipeline import STAGES, Pipeline
from driveguard.risk.collision import RiskLevel
from driveguard.timing import Stopwatch
from driveguard.video_io import VideoSource, VideoWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DriveGuard AI - lanes, departure warning, detection, tracking, collision risk")
    p.add_argument("--source", required=True, help="video path (or 0 for webcam)")
    p.add_argument("--output", default="outputs/annotated.mp4")
    p.add_argument("--events", default="outputs/events.jsonl", help="JSON Lines event log (alerts with frame + time)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--show", action="store_true", help="live preview window (press q to quit)")
    p.add_argument("--debug", action="store_true", help="draw lane ROI, edge-map inset and ego corridor")
    return p.parse_args()


def _announce(event: dict) -> None:
    """One console line per alert, as it happens."""
    kind, n = event["kind"], event["frame"]
    if kind == "lane_departure":
        print(f"  [frame {n}] LANE DEPARTURE ({event['side']}), offset {event['offset']:+.2f}", flush=True)
    elif kind == "collision_critical":
        print(f"  [frame {n}] COLLISION RISK CRITICAL: track #{event['track_id']}, TTC {event['ttc']:.1f}s", flush=True)
    else:
        data = {k: v for k, v in event.items() if k not in ("kind", "frame", "t")}
        print(f"  [frame {n}] {kind.upper().replace('_', ' ')}: {data}", flush=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    source = int(args.source) if args.source.isdigit() else args.source

    detector = Detector(cfg.detector)
    detector.warmup()
    events = EventLog(args.events)
    proc = psutil.Process()

    writer = None
    stages: dict[str, list[float]] = {k: [] for k in ("decode+resize", *STAGES, "draw+write")}
    seen_ids: set[int] = set()
    both_fresh = both_any = warning_frames = warning_episodes = 0
    risk_frames = {level: 0 for level in RiskLevel}   # frames whose worst risk was this level
    critical_events = 0
    peak_rss = 0.0
    fps_ema = 0.0
    n = 0
    t_start = time.perf_counter()

    with VideoSource(source, cfg.video.max_width) as video:
        pipeline = Pipeline(cfg, detector, fps=video.fps, events=events)
        frames = video.frames()
        while True:
            sw = Stopwatch()
            frame = next(frames, None)
            if frame is None:
                break
            stages["decode+resize"].append(sw.lap_ms())

            result = pipeline.process(frame)
            for stage in STAGES:
                stages[stage].append(result.timings_ms[stage])
            sw.lap_ms()   # restart the stopwatch: pipeline time is already recorded above

            lane, state = result.lane, result.departure
            both_fresh += lane.left_fresh and lane.right_fresh
            both_any += lane.left is not None and lane.right is not None
            warning_frames += state.warning
            warning_episodes += state.started
            seen_ids.update(t.track_id for t in result.tracks)
            risk_frames[max((r.level for r in result.risks.values()), default=RiskLevel.SAFE)] += 1
            for event in result.events:
                critical_events += event["kind"] == "collision_critical"
                _announce(event)

            pipeline.annotate(frame, result, fps=fps_ema, debug=args.debug)
            if writer is None:
                h, w = frame.shape[:2]
                writer = VideoWriter(args.output, video.fps, (w, h))
            writer.write(frame)

            quit_requested = False
            if args.show:
                cv2.imshow("DriveGuard AI", frame)
                quit_requested = (cv2.waitKey(1) & 0xFF) == ord("q")
            stages["draw+write"].append(sw.lap_ms())

            # full-loop FPS: sum of all stages, decode included
            loop_ms = sum(v[-1] for v in stages.values())
            inst = 1000.0 / loop_ms
            fps_ema = inst if n == 0 else 0.9 * fps_ema + 0.1 * inst

            n += 1
            if n % 25 == 0:
                peak_rss = max(peak_rss, proc.memory_info().rss / 1e6)
                print(f"  {n}/{video.total_frames or '?'} frames", flush=True)
            if quit_requested or (args.max_frames and n >= args.max_frames):
                break

    if writer:
        writer.release()
    events.close()
    cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    peak_rss = max(peak_rss, proc.memory_info().rss / 1e6)
    print("\n--- Summary ---")
    print(f"Frames processed : {n}")
    print(f"End-to-end FPS   : {n / elapsed:.1f}  (wall clock, everything included)")
    print("Per-stage latency, ms (avg / p95):")
    for name, vals in stages.items():
        if vals:
            print(f"  {name:<14} : {np.mean(vals):6.1f} / {np.percentile(vals, 95):6.1f}")
    print(f"Unique track IDs : {len(seen_ids)}")
    if n:
        print(f"Both lanes       : {100 * both_fresh / n:.1f}% measured this frame, "
              f"{100 * both_any / n:.1f}% incl. held estimates")
    print(f"Departure warns  : {warning_episodes} episode(s), {warning_frames} frame(s)")
    if n:
        print("Collision risk   : frames by worst level in view - "
              + ", ".join(f"{lv.name.lower()} {100 * c / n:.1f}%" for lv, c in risk_frames.items()))
    print(f"Critical events  : {critical_events} (distinct objects that reached CRITICAL)")
    print("Events logged    : " + (", ".join(f"{k} {v}" for k, v in sorted(events.counts.items())) or "none"))
    print(f"Process RAM      : {peak_rss:.0f} MB peak (sampled)")
    print(f"Output           : {args.output}")
    print(f"Event log        : {args.events}")


if __name__ == "__main__":
    main()
