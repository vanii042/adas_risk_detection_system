"""Lane-departure logic: lateral offset from the lane centre, with hysteresis."""
from dataclasses import dataclass

from driveguard.config import DepartureConfig
from driveguard.lane.lane_detector import LaneResult


@dataclass(frozen=True)
class DepartureState:
    offset: float | None   # (vehicle centre - lane centre) / lane width; > 0 means right of centre
    side: str | None       # "left" or "right" while a warning is active, else None
    warning: bool
    started: bool = False  # True only on the frame a new warning begins (used for event logging)


class DepartureChecker:
    """Turns per-frame lane estimates into a stable warning.

    - Both lines visible: offset is measured against the lane centre.
    - One line visible: the lane width remembered from earlier frames is used instead.
    - Hysteresis (separate warn/clear thresholds plus frame counts) stops the warning flickering.
    """

    def __init__(self, cfg: DepartureConfig) -> None:
        self._cfg = cfg
        self._lane_width: float | None = None  # pixels, at the ROI bottom row
        self._warning = False
        self._side: str | None = None
        self._above = 0
        self._below = 0
        self._blind = 0

    def update(self, lane: LaneResult, frame_width: int) -> DepartureState:
        c = self._cfg
        offset = self._offset(lane, frame_width)
        started = False

        if offset is None:
            self._blind += 1
            self._above = 0
            if self._warning and self._blind > c.clear_frames * 2:
                self._warning, self._side = False, None  # cannot verify the drift any more
        else:
            self._blind = 0
            magnitude = abs(offset)
            if not self._warning:
                self._above = self._above + 1 if magnitude >= c.warn_offset else 0
                if self._above >= c.trigger_frames:
                    self._warning, started, self._below = True, True, 0
            else:
                self._below = self._below + 1 if magnitude <= c.clear_offset else 0
                if self._below >= c.clear_frames:
                    self._warning, self._above = False, 0
            self._side = ("right" if offset > 0 else "left") if self._warning else None

        return DepartureState(offset, self._side if self._warning else None, self._warning, started)

    def _offset(self, lane: LaneResult, frame_width: int) -> float | None:
        c = self._cfg
        left, right = lane.left, lane.right

        if left and right:
            width = right.x_bottom - left.x_bottom
            if c.min_lane_width <= width / frame_width <= c.max_lane_width:
                a = c.width_smoothing
                self._lane_width = width if self._lane_width is None else (1 - a) * self._lane_width + a * width
                centre = (left.x_bottom + right.x_bottom) / 2
            else:
                return None  # implausible pair, e.g. markers from two different lanes
        elif self._lane_width is not None and left:
            centre = left.x_bottom + self._lane_width / 2
        elif self._lane_width is not None and right:
            centre = right.x_bottom - self._lane_width / 2
        else:
            return None

        vehicle_x = c.camera_center_x * frame_width
        return float((vehicle_x - centre) / self._lane_width)