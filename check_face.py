"""Steps 1-2 check - do the landmarks and skin patches sit on the face?

    python check_face.py data/mine/clip.avi --frame 100

Saves outputs/<video>_regions.png (face with patches drawn) and
outputs/<video>_patch_traces.png (each patch's mean green over time).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from pulse.extract import extract_traces
from pulse.plots import style
from pulse.roi import draw_regions
from pulse.video import FaceTracker, iter_frames, video_fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20)
    args = ap.parse_args()
    stem = Path(args.video).stem
    Path("outputs").mkdir(exist_ok=True)

    fps = video_fps(args.video)
    print(f"Frame rate reported by the file: {fps:.2f} fps")
    frame = next(iter_frames(args.video, 1, args.frame), None)
    if frame is None:
        raise SystemExit("Could not read that frame")
    with FaceTracker(video_mode=False) as t:
        lm = t.landmarks(frame)
    if lm is None:
        raise SystemExit("No face found in that frame - try another --frame")
    print(f"Found {len(lm)} landmarks")
    out = Path("outputs") / f"{stem}_regions.png"
    cv2.imwrite(str(out), cv2.cvtColor(draw_regions(frame, lm), cv2.COLOR_RGB2BGR))
    print(f"Saved {out.resolve()}")

    tr = extract_traces(args.video, max_frames=int(args.seconds * fps), verbose=True)
    plt = style()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    t = np.arange(tr.n_frames) / tr.fps
    for k, r in enumerate(tr.regions):
        g = tr.rgb[:, k, 1]
        ax.plot(t, g - g.mean(), label=r, lw=1.2)
    ax.set_title("Mean green per patch, minus its average (one number per frame per patch)")
    ax.set_xlabel("seconds")
    ax.set_ylabel("green level change")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    out2 = Path("outputs") / f"{stem}_patch_traces.png"
    fig.savefig(out2)
    print(f"Saved {out2.resolve()}  (face found in {100 * tr.face_found.mean():.0f}% of frames)")


if __name__ == "__main__":
    main()
