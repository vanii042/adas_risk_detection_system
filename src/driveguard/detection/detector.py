"""Thin wrapper around Ultralytics YOLO so the rest of the code never touches it directly."""
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from driveguard.config import DetectorConfig
from driveguard.detection.types import Detection  # re-exported: existing imports keep working

__all__ = ["Detection", "Detector"]


class Detector:
    def __init__(self, cfg: DetectorConfig) -> None:
        Path(cfg.model).parent.mkdir(parents=True, exist_ok=True)
        self._cfg = cfg
        self._model = YOLO(cfg.model)
        self._names = self._model.names

    @property
    def names(self) -> dict[int, str]:
        return self._names

    def warmup(self, shape: tuple[int, int, int] = (360, 640, 3)) -> None:
        """First inference is slow (lazy init); run one dummy frame so timings are fair."""
        self.detect(np.zeros(shape, dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self._model.predict(
            frame,
            imgsz=self._cfg.imgsz,
            conf=self._cfg.conf,
            iou=self._cfg.iou,
            classes=self._cfg.classes,
            device=self._cfg.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(tuple(float(v) for v in b), float(c), int(k), self._names[int(k)])
            for b, c, k in zip(xyxy, conf, cls)
        ]