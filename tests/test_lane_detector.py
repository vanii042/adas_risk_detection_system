import cv2
import numpy as np

from driveguard.config import LaneConfig
from driveguard.lane.lane_detector import LaneDetector


def _road_frame(paint: int = 255) -> np.ndarray:
    """Synthetic road: two lane lines converging towards the horizon."""
    frame = np.full((360, 640, 3), 60, dtype=np.uint8)
    colour = (paint, paint, paint)
    cv2.line(frame, (110, 360), (290, 200), colour, 6)
    cv2.line(frame, (530, 360), (350, 200), colour, 6)
    return frame


def _blank_frame() -> np.ndarray:
    return np.full((360, 640, 3), 60, dtype=np.uint8)


def _cfg() -> LaneConfig:
    # the synthetic road spans the lower-middle of the frame, so use a tall ROI
    return LaneConfig(roi_top_y=0.55, roi_bottom_y=0.98, roi_bottom_half_width=0.48)


def test_detects_both_lane_lines():
    detector = LaneDetector(_cfg())
    for _ in range(10):
        lane = detector.detect(_road_frame())
    assert lane.left is not None and lane.right is not None
    assert lane.left_fresh and lane.right_fresh
    assert lane.left.x_bottom < lane.right.x_bottom


def test_detects_dull_paint_that_a_fixed_threshold_would_miss():
    detector = LaneDetector(_cfg())
    for _ in range(10):
        lane = detector.detect(_road_frame(paint=130))  # well below white_min_lightness
    assert lane.left is not None and lane.right is not None


def test_blank_road_gives_no_lanes():
    lane = LaneDetector(_cfg()).detect(_blank_frame())
    assert lane.left is None and lane.right is None


def test_lane_is_held_briefly_then_dropped_and_marked_not_fresh():
    cfg = _cfg()
    detector = LaneDetector(cfg)
    for _ in range(5):
        detector.detect(_road_frame())
    for _ in range(cfg.hold_frames):
        lane = detector.detect(_blank_frame())
        assert lane.left is not None and not lane.left_fresh
    lane = detector.detect(_blank_frame())
    assert lane.left is None and lane.right is None