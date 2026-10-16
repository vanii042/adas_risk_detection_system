"""Sanity-check the DriveGuard environment."""
import platform
import sys

import psutil


def main() -> None:
    mem = psutil.virtual_memory()
    print(f"Python      : {sys.version.split()[0]} on {platform.system()}")
    print(f"RAM         : {mem.total / 1e9:.1f} GB total, {mem.available / 1e9:.1f} GB free")

    import cv2
    import numpy
    import supervision
    import torch
    import ultralytics

    print(f"NumPy       : {numpy.__version__}")
    print(f"OpenCV      : {cv2.__version__}")
    print(f"PyTorch     : {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    print(f"Ultralytics : {ultralytics.__version__}")
    print(f"Supervision : {supervision.__version__}")

    x = torch.rand(3, 3) @ torch.rand(3, 3)
    print(f"Torch matmul OK, result shape {tuple(x.shape)}")

    import driveguard  # noqa: F401
    print("driveguard package imports OK")


if __name__ == "__main__":
    main()