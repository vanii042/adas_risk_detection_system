"""Classical lane detection: lane-paint mask + Canny + Hough, with temporal smoothing."""
from dataclasses import dataclass

import cv2
import numpy as np

from driveguard.config import LaneConfig


@dataclass(frozen=True)
class LaneLine:
    """A lane boundary, described by its x position at the ROI's bottom and top rows."""

    x_bottom: float
    x_top: float


@dataclass(frozen=True)
class LaneResult:
    left: LaneLine | None
    right: LaneLine | None
    y_bottom: int
    y_top: int
    left_fresh: bool = False   # True if measured in this frame, False if held from earlier frames
    right_fresh: bool = False


def _at_least(img: np.ndarray, value: int) -> np.ndarray:
    """Binary mask (0/255) of pixels >= value, for a single-channel uint8 image."""
    _, mask = cv2.threshold(img, int(value) - 1, 255, cv2.THRESH_BINARY)
    return mask


class LaneDetector:
    def __init__(self, cfg: LaneConfig) -> None:
        self._cfg = cfg
        self._dilate_kernel = np.ones((5, 5), np.uint8)
        k = cfg.tophat_kernel | 1  # odd size
        self._tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._roi_cache: tuple[tuple[int, int], np.ndarray, np.ndarray] | None = None
        self._smoothed: dict[str, LaneLine | None] = {"left": None, "right": None}
        self._missed = {"left": 0, "right": 0}
        # intermediate results, kept for the debug overlay and scripts/tune_lane.py
        self.debug_paint: np.ndarray | None = None
        self.debug_canny: np.ndarray | None = None
        self.debug_edges: np.ndarray | None = None
        self.debug_segments: list[tuple[float, float, float, float]] = []

    def roi_polygon(self, shape: tuple[int, int]) -> np.ndarray:
        return self._roi(shape)[0]

    def roi_mask(self, shape: tuple[int, int]) -> np.ndarray:
        return self._roi(shape)[1]

    def detect(self, frame: np.ndarray) -> LaneResult:
        h, w = frame.shape[:2]
        c = self._cfg
        y_bottom, y_top = int(h * c.roi_bottom_y), int(h * c.roi_top_y)

        edges = self._edges(frame)
        segments = self._segments(edges)
        self.debug_edges = edges
        self.debug_segments = segments

        left, left_fresh = self._smooth("left", self._measure(segments, "left", w, y_bottom, y_top))
        right, right_fresh = self._smooth("right", self._measure(segments, "right", w, y_bottom, y_top))
        return LaneResult(left, right, y_bottom, y_top, left_fresh, right_fresh)

    # ------------------------------------------------------------------ internals
    def _roi(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if self._roi_cache is None or self._roi_cache[0] != shape:
            h, w = shape
            c = self._cfg
            cx = w / 2
            polygon = np.array(
                [
                    (cx - c.roi_bottom_half_width * w, c.roi_bottom_y * h),
                    (cx - c.roi_top_half_width * w, c.roi_top_y * h),
                    (cx + c.roi_top_half_width * w, c.roi_top_y * h),
                    (cx + c.roi_bottom_half_width * w, c.roi_bottom_y * h),
                ],
                dtype=np.int32,
            )
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            self._roi_cache = (shape, polygon, mask)
        return self._roi_cache[1], self._roi_cache[2]

    def _paint_mask(self, blur: np.ndarray) -> np.ndarray:
        c = self._cfg
        hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        light = hls[:, :, 1]

        # 1) very bright paint, 2) yellow paint, 3) paint brighter than the road around it.
        # The top-hat term is what keeps dull paint (dusk, shade, worn markings) detectable.
        white = _at_least(light, c.white_min_lightness)
        yellow = cv2.inRange(hls, np.array([15, 60, 90]), np.array([35, 255, 255]))
        tophat = cv2.morphologyEx(light, cv2.MORPH_TOPHAT, self._tophat_kernel)
        local = cv2.bitwise_and(_at_least(tophat, c.tophat_min), _at_least(light, c.min_paint_lightness))
        paint = cv2.bitwise_or(cv2.bitwise_or(white, yellow), local)
        return cv2.dilate(paint, self._dilate_kernel)

    def _edges(self, frame: np.ndarray) -> np.ndarray:
        c = self._cfg
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        paint = self._paint_mask(blur)
        canny = cv2.Canny(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), c.canny_low, c.canny_high)
        self.debug_paint = paint
        self.debug_canny = canny

        edges = cv2.bitwise_and(canny, paint)  # keep only edges of lane-coloured pixels
        return cv2.bitwise_and(edges, self.roi_mask(frame.shape[:2]))

    def _segments(self, edges: np.ndarray) -> list[tuple[float, float, float, float]]:
        c = self._cfg
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, c.hough_threshold,
            minLineLength=c.min_line_length, maxLineGap=c.max_line_gap,
        )
        if lines is None:
            return []
        # OpenCV 4 returns (N, 1, 4), OpenCV 5 returns (N, 4); reshape handles both.
        return [tuple(float(v) for v in seg) for seg in np.asarray(lines).reshape(-1, 4)]

    def _measure(self, segments, side: str, w: int, y_bottom: int, y_top: int) -> LaneLine | None:
        c = self._cfg
        cx = w / 2
        candidates = []
        for x1, y1, x2, y2 in segments:
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            slope = dy / dx  # image y grows downward: left markers slope negative, right positive
            if not c.min_abs_slope <= abs(slope) <= c.max_abs_slope:
                continue
            if (slope < 0) != (side == "left"):
                continue
            mid_x = (x1 + x2) / 2
            on_correct_side = mid_x < cx if side == "left" else mid_x > cx
            if not on_correct_side:
                continue
            x_at_bottom = x1 + (y_bottom - y1) / slope
            candidates.append((x_at_bottom, x1, y1, x2, y2))
        if not candidates:
            return None

        # On multi-lane roads keep only the marker closest to the vehicle's centre line.
        ref = min(candidates, key=lambda s: abs(s[0] - cx))[0]
        keep = [s for s in candidates if abs(s[0] - ref) <= c.cluster_tol * w]

        ys, xs, weights = [], [], []
        for _, x1, y1, x2, y2 in keep:
            length = float(np.hypot(x2 - x1, y2 - y1))
            ys += [y1, y2]
            xs += [x1, x2]
            weights += [length, length]
        if len(set(ys)) < 2:
            return None
        m, b = np.polyfit(ys, xs, 1, w=weights)  # x as a function of y: stable for near-vertical lines
        return LaneLine(float(m * y_bottom + b), float(m * y_top + b))

    def _smooth(self, side: str, measurement: LaneLine | None) -> tuple[LaneLine | None, bool]:
        """Blend new measurements into the running estimate; hold it briefly when paint is missing."""
        prev = self._smoothed[side]
        if measurement is None:
            self._missed[side] += 1
            if self._missed[side] > self._cfg.hold_frames:
                self._smoothed[side] = None
            return self._smoothed[side], False

        self._missed[side] = 0
        a = self._cfg.smoothing
        if prev is None:
            self._smoothed[side] = measurement
        else:
            self._smoothed[side] = LaneLine(
                prev.x_bottom + a * (measurement.x_bottom - prev.x_bottom),
                prev.x_top + a * (measurement.x_top - prev.x_top),
            )
        return self._smoothed[side], True