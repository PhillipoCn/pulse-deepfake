"""Step 5 on a dataset - turn FaceForensics++ videos into labelled feature windows.

    python build_ffpp.py data/ffpp --methods Deepfakes FaceSwap --limit 200 --workers 6

Expects the standard FaceForensics++ layout:
  <root>/original_sequences/youtube/<c23>/videos/000.mp4                 (real)
  <root>/manipulated_sequences/<Method>/<c23>/videos/000_003.mp4         (fake)
A fake named 000_003 is made from real videos 000 and 003, so all three share a group:
they must stay on the same side of the train/test split (step 7).
Output: outputs/ffpp_features.npz (X = one row per 6 s window, V = one row per video)
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pulse.detail import VIDEO_FEATURE_NAMES

ALL_METHODS = ["Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures", "FaceShifter"]


def list_videos(root, compression, methods, limit):
    root = Path(root)
    items = []
    real_dir = root / "original_sequences" / "youtube" / compression / "videos"
    reals = sorted(real_dir.glob("*.mp4"))[:limit]
    if not reals:
        raise SystemExit(f"No real videos found in {real_dir}")
    items += [(p, 0, "real") for p in reals]
    for m in methods:
        d = root / "manipulated_sequences" / m / compression / "videos"
        fakes = sorted(d.glob("*.mp4"))[:limit]
        if not fakes:
            print(f"warning: no videos in {d}")
        items += [(p, 1, m) for p in fakes]
    return items


def group_ids(stems):
    """Union-find over identity numbers: '000_003' links 000 and 003."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s in stems:
        ids = s.split("_")
        for other in ids[1:]:
            parent[find(ids[0])] = find(other)
        find(ids[0])
    roots = {}
    return [roots.setdefault(find(s.split("_")[0]), len(roots)) for s in stems]


def _work(path, seconds, win_s, step_s):
    # Imported here so each worker process loads MediaPipe itself.
    os.environ.setdefault("GLOG_minloglevel", "2")
    from pulse.extract import extract_traces
    from pulse.detail import video_vector
    from pulse.video import video_fps
    fps = video_fps(path)
    tr = extract_traces(path, max_frames=int(seconds * fps))
    if tr.face_found.mean() < 0.8:
        raise ValueError(f"face found in only {100 * tr.face_found.mean():.0f}% of frames")
    vec, windows = video_vector(tr, win_s, step_s)
    return windows, vec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    ap.add_argument("--methods", nargs="+", default=["Deepfakes", "FaceSwap"], choices=ALL_METHODS)
    ap.add_argument("--limit", type=int, default=None, help="max videos per category")
    ap.add_argument("--seconds", type=float, default=20, help="use the first N seconds of each video")
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--step", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default="outputs/ffpp_features.npz")
    args = ap.parse_args()

    items = list_videos(args.root, args.compression, args.methods, args.limit)
    print(f"{len(items)} videos ({sum(1 for i in items if i[1] == 0)} real) - {args.workers} workers")
    X, y, vid, cat, stems, V = [], [], [], [], [], []
    failed = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_work, str(p), args.seconds, args.window, args.step): (p, lab, c) for p, lab, c in items}
        for k, f in enumerate(as_completed(futs), 1):
            p, lab, c = futs[f]
            try:
                feats, vec = f.result()
            except Exception as e:
                failed += 1
                print(f"  skip {c}/{p.name}: {e}")
                continue
            v = len(stems)
            stems.append(p.stem)
            V.append(vec)
            for row in feats:
                X.append(row); y.append(lab); vid.append(v); cat.append(c)
            if k % 25 == 0 or k == len(items):
                print(f"  {k}/{len(items)} done", flush=True)
    groups_per_video = np.array(group_ids(stems))
    vid = np.array(vid)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=np.array(X), y=np.array(y), video=vid, groups=groups_per_video[vid],
             category=np.array(cat), stems=np.array(stems),
             V=np.array(V), video_feature_names=np.array(VIDEO_FEATURE_NAMES),
             window=args.window, step=args.step, seconds=args.seconds)
    print(f"Saved {len(X)} windows from {len(stems)} videos ({failed} skipped) "
          f"in {len(set(groups_per_video))} identity groups -> {out.resolve()}")


if __name__ == "__main__":
    main()
