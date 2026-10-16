"""Turn an annotated output video into a small looping GIF for the README.

Usage (from project root):
    python scripts/make_demo_gif.py --input outputs/annotated.mp4 --output docs/images/demo.gif
"""
import argparse

import cv2
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser(description="Annotated video -> README demo GIF")
    p.add_argument("--input", default="outputs/annotated.mp4")
    p.add_argument("--output", default="docs/images/demo.gif")
    p.add_argument("--width", type=int, default=480, help="GIF width in px (smaller = lighter file)")
    p.add_argument("--fps", type=float, default=12.0, help="GIF frame rate")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--seconds", type=float, default=8.0, help="clip length in seconds")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / args.fps)), 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * src_fps))
    last = int((args.start + args.seconds) * src_fps)

    frames, i = [], int(args.start * src_fps)
    while i < last:
        ok, frame = cap.read()
        if not ok:
            break
        if (i - int(args.start * src_fps)) % step == 0:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (args.width, int(h * args.width / w)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    if not frames:
        raise SystemExit("No frames read; check --start/--seconds")

    frames[0].save(args.output, save_all=True, append_images=frames[1:], loop=0,
                   duration=int(1000 / args.fps), optimize=True)
    print(f"Wrote {args.output}: {len(frames)} frames")


if __name__ == "__main__":
    main()
