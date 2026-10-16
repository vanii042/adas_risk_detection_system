"""Print basic info for every video in data/samples."""
from pathlib import Path

import cv2


def main() -> None:
    clips = sorted(Path("data/samples").glob("*.mp4"))
    if not clips:
        print("No .mp4 files found in data/samples")
        return
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        ok, _ = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{clip.name}: {w}x{h}, {fps:.1f} fps, {frames} frames, readable={ok}")
        cap.release()


if __name__ == "__main__":
    main()