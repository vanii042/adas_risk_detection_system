"""Frame reading and video writing helpers."""
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def resize_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale to max_width keeping aspect ratio; never upscale."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


class VideoSource:
    """Streams frames one at a time so memory use stays flat."""

    def __init__(self, source: str | Path | int, max_width: int = 640) -> None:
        self._cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self._max_width = max_width
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield resize_to_width(frame, self._max_width)

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # getattr keeps this working across OpenCV 4.x and 5.x
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
        self._writer = cv2.VideoWriter(str(path), fourcc_fn(*"mp4v"), fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for: {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()