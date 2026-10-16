import numpy as np

from driveguard.video_io import resize_to_width


def test_resize_keeps_aspect_ratio():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert resize_to_width(frame, 640).shape == (360, 640, 3)


def test_resize_never_upscales():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert resize_to_width(frame, 640).shape == frame.shape