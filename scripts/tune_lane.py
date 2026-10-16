"""Save a 4-panel diagnostic image of the lane pipeline for a single frame.

Usage (from project root, venv active):
    python scripts\\tune_lane.py --source data\\samples\\urban_busy.mp4 --frame 60 --config configs\\urban.yaml
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from driveguard.config import load_config
from driveguard.lane.lane_detector import LaneDetector
from driveguard.video_io import VideoSource
from driveguard.viz.overlay import draw_lane


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DriveGuard AI - lane pipeline diagnostic image")
    p.add_argument("--source", required=True)
    p.add_argument("--frame", type=int, default=60, help="frame index to inspect")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output", default="outputs/tune_lane.png")
    return p.parse_args()


def _panel(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (len(text) * 9 + 10, 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    detector = LaneDetector(cfg.lane)

    frame, lane = None, None
    with VideoSource(args.source, cfg.video.max_width) as video:
        for i, f in enumerate(video.frames()):
            if i >= args.frame - 15:  # warm up smoothing on the 15 frames before the target
                lane = detector.detect(f)
            if i >= args.frame:
                frame = f
                break
    if frame is None:
        raise SystemExit(f"Video has fewer than {args.frame + 1} frames")

    shape = frame.shape[:2]
    roi_mask = detector.roi_mask(shape)

    p1 = frame.copy()
    draw_lane(p1, lane)
    cv2.polylines(p1, [detector.roi_polygon(shape)], True, (0, 255, 255), 1, cv2.LINE_AA)

    p2 = cv2.bitwise_and(detector.debug_paint, detector.debug_paint, mask=roi_mask)
    p3 = detector.debug_canny
    p4 = cv2.cvtColor(cv2.dilate(detector.debug_edges, np.ones((3, 3), np.uint8)), cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in detector.debug_segments:
        cv2.line(p4, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_AA)

    grid = np.vstack([
        np.hstack([_panel(p1, "frame + ROI + lane"), _panel(p2, "paint mask inside ROI")]),
        np.hstack([_panel(p3, "Canny edges (everywhere)"), _panel(p4, "final edges + Hough segments")]),
    ])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, grid)

    print(f"Paint pixels in ROI : {cv2.countNonZero(cv2.bitwise_and(detector.debug_paint, roi_mask))}")
    print(f"Edge pixels (final) : {cv2.countNonZero(detector.debug_edges)}")
    print(f"Hough segments      : {len(detector.debug_segments)}")
    print(f"Lanes found         : left={lane.left is not None}, right={lane.right is not None}")
    print(f"Saved               : {args.output}")


if __name__ == "__main__":
    main()