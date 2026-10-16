"""Plain data types for detections. Kept free of heavy imports (torch, ultralytics)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str
