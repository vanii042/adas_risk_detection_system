"""Typed configuration loaded from YAML."""
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class VideoConfig:
    max_width: int = 640


@dataclass
class DetectorConfig:
    model: str = "models/yolo11n.pt"
    imgsz: int = 416
    conf: float = 0.35
    iou: float = 0.5
    device: str = "cpu"
    classes: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7, 9, 11])


@dataclass
class TrackerConfig:
    lost_track_buffer: int = 30


@dataclass
class LaneConfig:
    # region of interest: a trapezoid in front of the car (fractions of frame size).
    # Camera-specific: bottom edge must sit above the bonnet, top edge below the horizon.
    roi_top_y: float = 0.55
    roi_bottom_y: float = 0.72
    roi_top_half_width: float = 0.12
    roi_bottom_half_width: float = 0.45
    # lane-paint detection: bright paint (absolute) OR paint brighter than its surroundings (top-hat)
    white_min_lightness: int = 190
    tophat_kernel: int = 25
    tophat_min: int = 28
    min_paint_lightness: int = 90
    # edges and line finding
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 15
    min_line_length: int = 15
    max_line_gap: int = 100
    min_abs_slope: float = 0.4
    max_abs_slope: float = 2.5
    cluster_tol: float = 0.07
    # temporal behaviour
    smoothing: float = 0.25
    hold_frames: int = 20


@dataclass
class DepartureConfig:
    camera_center_x: float = 0.5   # where the vehicle's centre line sits in the frame (fraction of width)
    warn_offset: float = 0.30      # |offset| (fraction of lane width) that starts a warning
    clear_offset: float = 0.22     # |offset| below which a warning may clear (hysteresis)
    trigger_frames: int = 4        # consecutive frames above warn_offset before warning
    clear_frames: int = 8          # consecutive frames below clear_offset before clearing
    min_lane_width: float = 0.12   # plausible lane width at the ROI bottom (fraction of frame width)
    max_lane_width: float = 0.90
    width_smoothing: float = 0.05


@dataclass
class RiskConfig:
    # Ego corridor: trapezoid in front of the car (fractions of frame size). Camera-specific:
    # top edge sits near the horizon, bottom edge near the bonnet. Widths are half-widths.
    corridor_top_y: float = 0.50
    corridor_bottom_y: float = 0.80
    corridor_top_half_width: float = 0.06
    corridor_bottom_half_width: float = 0.30
    # time-to-collision proxy from bounding-box growth
    window: int = 8                # frames used for the growth fit
    min_history: int = 4           # frames needed before a TTC is reported
    min_slope: float = 0.02        # ignore growth slower than this (1/s): not approaching
    ttc_caution: float = 5.0       # seconds
    ttc_warning: float = 3.0
    ttc_critical: float = 1.5
    stale_after: int = 30          # frames a track may go unseen before its history is dropped
    confirm_frames: int = 3        # a level must hold this many consecutive frames before it is reported
    classes: list[str] = field(
        default_factory=lambda: ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
    )


@dataclass
class SignsConfig:
    enabled: bool = True
    # traffic-light colour classification (HSV vote inside the detected box)
    light_min_size: int = 8          # px; boxes smaller than this (either side) are too small to classify
    light_sat_min: int = 90          # minimum saturation for a pixel to count as coloured
    light_val_min: int = 140         # minimum brightness: lit lamps are bright, unlit ones are not
    light_min_fraction: float = 0.01 # share of the box a colour must cover to win
    vote_frames: int = 7             # per-track voting window (removes flicker)
    min_votes: int = 3               # identical votes needed inside the window to report a colour
    # which light counts as "the one ahead": near the centre line and in the upper part of the frame
    ahead_half_width: float = 0.25   # |light centre - frame centre| as a fraction of frame width
    ahead_max_y: float = 0.60        # light centre must be above this fraction of frame height
    # stop sign: alert once the sign is at least this tall (fraction of frame height), i.e. close enough
    stop_min_height: float = 0.08
    stale_after: int = 30            # frames a track may go unseen before its state is dropped


@dataclass
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    lane: LaneConfig = field(default_factory=LaneConfig)
    departure: DepartureConfig = field(default_factory=DepartureConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    signs: SignsConfig = field(default_factory=SignsConfig)


class ConfigError(ValueError):
    """The YAML config has an unknown section/key or a malformed section."""


def _section(cls, raw: dict, name: str):
    """Build one config dataclass, rejecting unknown keys (usually typos) with a readable message."""
    values = raw.get(name) or {}
    if not isinstance(values, dict):
        raise ConfigError(f"'{name}' must be a mapping of settings, got {type(values).__name__}")
    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ConfigError(
            f"unknown setting(s) in '{name}': {', '.join(unknown)}. Valid settings: {', '.join(sorted(valid))}"
        )
    return cls(**values)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping of sections")
    sections = {f.name for f in fields(Config)}
    unknown = sorted(set(raw) - sections)
    if unknown:
        raise ConfigError(f"{path}: unknown section(s): {', '.join(unknown)}. Valid sections: {', '.join(sorted(sections))}")
    return Config(
        video=_section(VideoConfig, raw, "video"),
        detector=_section(DetectorConfig, raw, "detector"),
        tracker=_section(TrackerConfig, raw, "tracker"),
        lane=_section(LaneConfig, raw, "lane"),
        departure=_section(DepartureConfig, raw, "departure"),
        risk=_section(RiskConfig, raw, "risk"),
        signs=_section(SignsConfig, raw, "signs"),
    )