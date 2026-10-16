"""Drawing helpers. Pure functions: take a frame, draw on it, return it."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:  # only needed for type hints; keeps this module light to import
    from driveguard.detection.types import Detection
    from driveguard.lane.departure import DepartureState
    from driveguard.lane.lane_detector import LaneResult
    from driveguard.risk.collision import RiskResult
    from driveguard.signs.traffic import SignalReport
    from driveguard.tracking.tracker import Track

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PALETTE = [
    (66, 135, 245), (80, 200, 120), (245, 166, 35), (200, 90, 220),
    (60, 190, 230), (90, 90, 240), (170, 170, 60), (230, 120, 120),
]
_GREEN, _AMBER, _RED = (0, 200, 0), (0, 190, 255), (0, 0, 230)  # BGR
_ORANGE = (0, 120, 255)
_RISK_COLORS = {1: _AMBER, 2: _ORANGE, 3: _RED}   # RiskLevel.CAUTION / WARNING / CRITICAL
_RISK_NAMES = {1: "CAUTION", 2: "WARNING", 3: "CRITICAL"}
_LIGHT_COLORS = {"red": _RED, "yellow": _AMBER, "green": _GREEN}   # LightState.value -> BGR


def _draw_box(frame: np.ndarray, xyxy: tuple, color: tuple, text: str) -> None:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), base = cv2.getTextSize(text, _FONT, 0.5, 1)
    top = max(y1 - th - base - 4, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 4, top + th + base + 4), color, -1)
    cv2.putText(frame, text, (x1 + 2, top + th + 1), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for det in detections:
        color = _PALETTE[det.class_id % len(_PALETTE)]
        _draw_box(frame, det.xyxy, color, f"{det.label} {det.confidence:.2f}")
    return frame


def draw_tracks(frame: np.ndarray, tracks: list[Track],
                risks: dict[int, RiskResult] | None = None) -> np.ndarray:
    """Box colour follows the object, unless it carries collision risk: then it follows the risk level."""
    for t in tracks:
        color = _PALETTE[t.track_id % len(_PALETTE)]  # colour follows the object, not the class
        text = f"#{t.track_id} {t.label} {t.confidence:.2f}"
        risk = risks.get(t.track_id) if risks else None
        if risk and risk.level > 0:
            color = _RISK_COLORS[int(risk.level)]
            text = f"#{t.track_id} {t.label} {_RISK_NAMES[int(risk.level)]}"
            if risk.ttc is not None:
                text += f" {risk.ttc:.1f}s"
        _draw_box(frame, t.xyxy, color, text)
    return frame


def draw_corridor(frame: np.ndarray, polygon: list[tuple[int, int]]) -> np.ndarray:
    """Debug aid: the ego corridor in which objects count as 'in path' (cyan)."""
    cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], True, (255, 255, 0), 1, cv2.LINE_AA)
    return frame


def draw_collision_warning(frame: np.ndarray, risks: dict[int, RiskResult]) -> np.ndarray:
    """Banner (bottom centre) for the worst risk in view; nothing is drawn when everything is safe."""
    worst = max(risks.values(), key=lambda r: (r.level, -(r.ttc or 1e9)), default=None)
    if worst is None or worst.level < 2:   # caution is box-colour only; banner from WARNING up
        return frame
    h, w = frame.shape[:2]
    text = "COLLISION RISK" if worst.level == 2 else "BRAKE - COLLISION RISK"
    (tw, th), base = cv2.getTextSize(text, _FONT, 0.7, 2)
    x, y = (w - tw) // 2, h - 20
    cv2.rectangle(frame, (x - 10, y - th - 8), (x + tw + 10, y + base + 6), _RISK_COLORS[int(worst.level)], -1)
    cv2.putText(frame, text, (x, y), _FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_lane(frame: np.ndarray, lane: LaneResult, warning: bool = False) -> np.ndarray:
    """Green line = measured this frame, amber = held from earlier frames, red = departure warning."""
    w = frame.shape[1]

    def px(x: float) -> int:
        return int(np.clip(x, -5 * w, 6 * w))

    yb, yt = lane.y_bottom, lane.y_top
    if lane.left and lane.right:
        poly = np.array(
            [[px(lane.left.x_bottom), yb], [px(lane.left.x_top), yt],
             [px(lane.right.x_top), yt], [px(lane.right.x_bottom), yb]],
            dtype=np.int32,
        )
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], _RED if warning else _GREEN)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
    for line, fresh in ((lane.left, lane.left_fresh), (lane.right, lane.right_fresh)):
        if line:
            color = _RED if warning else (_GREEN if fresh else _AMBER)
            cv2.line(frame, (px(line.x_bottom), yb), (px(line.x_top), yt), color, 3, cv2.LINE_AA)
    return frame


def draw_departure(frame: np.ndarray, state: DepartureState) -> np.ndarray:
    """Lane-offset read-out (top-left, under the HUD) and a red banner while a warning is active."""
    h, w = frame.shape[:2]
    if state.offset is not None:
        cv2.rectangle(frame, (0, 26), (230, 48), (0, 0, 0), -1)
        cv2.putText(frame, f"lane offset {state.offset:+.2f}", (6, 42), _FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if state.warning and state.side:
        text = f"LANE DEPARTURE - {state.side.upper()}"
        (tw, th), base = cv2.getTextSize(text, _FONT, 0.6, 2)
        x, y = (w - tw) // 2, 34
        cv2.rectangle(frame, (x - 10, y - th - 8), (x + tw + 10, y + base + 6), _RED, -1)
        cv2.putText(frame, text, (x, y), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_debug(frame: np.ndarray, edges: np.ndarray, roi_polygon: np.ndarray) -> np.ndarray:
    """Yellow trapezoid = region of interest; top-right inset = edge map fed to the Hough transform."""
    cv2.polylines(frame, [roi_polygon], True, (0, 255, 255), 1, cv2.LINE_AA)
    h, w = frame.shape[:2]
    iw, ih = w // 4, h // 4
    small = cv2.resize(edges, (iw, ih), interpolation=cv2.INTER_AREA)
    _, small = cv2.threshold(small, 8, 255, cv2.THRESH_BINARY)
    frame[0:ih, w - iw:w] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(frame, (w - iw, 0), (w - 1, ih - 1), (0, 255, 255), 1)
    return frame


def draw_hud(frame: np.ndarray, fps: float, n_tracks: int) -> np.ndarray:
    text = f"FPS {fps:4.1f} | tracks {n_tracks}"
    cv2.rectangle(frame, (0, 0), (230, 26), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, 18), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def draw_signals(frame: np.ndarray, report: SignalReport, tracks: list[Track]) -> np.ndarray:
    """Traffic-light badge (left, under the lane read-out), a highlight on the light judged to be
    ours, and a banner while a stop sign is close. Nothing is drawn when no sign or light is in view."""
    h, w = frame.shape[:2]
    if report.ahead_id is not None:
        color = _LIGHT_COLORS.get(report.ahead_state.value, (110, 110, 110))
        text = f"LIGHT: {report.ahead_state.value.upper()}"
        cv2.rectangle(frame, (0, 48), (230, 70), color, -1)
        cv2.putText(frame, text, (6, 64), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for t in tracks:
            if t.track_id == report.ahead_id:
                x1, y1, x2, y2 = (int(v) for v in t.xyxy)
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 2)
    if report.stop_id is not None:
        text = "STOP SIGN AHEAD"
        (tw, th), base = cv2.getTextSize(text, _FONT, 0.6, 2)
        x, y = (w - tw) // 2, 70
        cv2.rectangle(frame, (x - 10, y - th - 8), (x + tw + 10, y + base + 6), _RED, -1)
        cv2.putText(frame, text, (x, y), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
