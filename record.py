"""Record a webcam clip for testing (lossless, so compression doesn't hide the pulse).

    python record.py --seconds 30

Tips: sit still, face a window or lamp (even light, no flicker), no backlight, and
count your pulse at the wrist (or read an oximeter) while recording.
Webcams often deliver fewer frames per second than they claim, so the real rate is
measured and saved next to the video as <name>.fps.json, which the pipeline reads.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open the camera. On macOS allow camera access for your terminal app in "
                         "System Settings > Privacy & Security > Camera, then restart the terminal.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Warming up the camera (3 s) - let auto-exposure settle...")
    t0 = time.time()
    while time.time() - t0 < 3:
        ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera returned no frames")
    h, w = frame.shape[:2]
    out = Path(args.out or f"data/mine/clip_{datetime.now():%Y%m%d_%H%M%S}.avi")
    out.parent.mkdir(parents=True, exist_ok=True)
    claimed = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), claimed, (w, h))

    print(f"Recording {args.seconds:.0f} s at {w}x{h}. Sit still and count your pulse now.")
    n, t0 = 0, time.time()
    while time.time() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
    elapsed = time.time() - t0
    writer.release()
    cap.release()
    real = n / elapsed
    json.dump({"fps": real}, open(str(out) + ".fps.json", "w"))
    print(f"Saved {out.resolve()}: {n} frames in {elapsed:.1f} s = {real:.2f} fps (camera claimed {claimed:.0f})")
    print(f"Next: python heart_rate.py {out} --true-bpm <your count>")


if __name__ == "__main__":
    main()
