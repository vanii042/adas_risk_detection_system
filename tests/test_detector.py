import numpy as np

from driveguard.config import load_config
from driveguard.detection.detector import Detector


def test_detector_returns_empty_on_blank_frame():
    detector = Detector(load_config().detector)
    assert detector.detect(np.zeros((360, 640, 3), dtype=np.uint8)) == []