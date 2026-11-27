"""Steps 1+2 together: video -> per-frame mean RGB of every skin patch.

Result: traces of shape (T, R, 3) = frames x regions x (R, G, B). Landmark detection is
the slow part, so results are cached as .npz files keyed by the video path.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .roi import DETECTOR_REGIONS, head_pose, region_mean_rgb
from .video import FaceTracker, iter_frames, video_fps

CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "cache"


@dataclass
class Traces:
    fps: float
    rgb: np.ndarray          # (T, R, 3) float
    regions: list
    face_found: np.ndarray   # (T,) bool
    pose: np.ndarray = None  # (T, 5) nose x, nose y, face width, yaw, pitch

    @property
    def n_frames(self):
        return self.rgb.shape[0]

    @property
    def duration(self):
        return self.n_frames / self.fps

    def region(self, name):
        return self.rgb[:, self.regions.index(name), :]


def _fill_gaps(x):
    """Linearly interpolate NaNs (frames where the face was lost) along time."""
    x = x.copy()
    t = np.arange(len(x))
    for idx in np.ndindex(x.shape[1:]):
        col = x[(slice(None),) + idx]
        bad = np.isnan(col)
        if bad.all():
            continue
        if bad.any():
            col[bad] = np.interp(t[bad], t[~bad], col[~bad])
    return x


def _remove_spikes(x, thresh=6.0):
    """Mark single-frame glitches (tracker jumps, the first-frame detection) as NaN.
    A glitch jumps away from BOTH neighbours in opposite directions by more than
    `thresh` robust std-devs of the frame-to-frame change. The smooth pulse never does."""
    x = x.copy()
    for idx in np.ndindex(x.shape[1:]):
        sl = (slice(None),) + idx
        col = x[sl]
        d = np.diff(col)
        good = np.isfinite(d)
        if good.sum() < 10:
            continue
        s = np.median(np.abs(d[good] - np.median(d[good]))) * 1.4826 + 1e-9
        lim = thresh * s
        bad = np.zeros(len(col), bool)
        with np.errstate(invalid="ignore"):
            bad[1:-1] = (np.abs(d[:-1]) > lim) & (np.abs(d[1:]) > lim) & (np.sign(d[:-1]) != np.sign(d[1:]))
            bad[0] = np.abs(d[0]) > lim
            bad[-1] = np.abs(d[-1]) > lim
        col[bad] = np.nan
        x[sl] = col
    return x


def extract_traces(path, max_frames=None, start_frame=0, regions=None, use_cache=True, verbose=False):
    path = Path(path)
    regions = list(regions or DETECTOR_REGIONS)
    key = hashlib.md5(f"v2|{path.resolve()}|{path.stat().st_size}|{max_frames}|{start_frame}|{regions}".encode()).hexdigest()
    cache = CACHE_DIR / f"{path.stem}_{key[:10]}.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=False)
        return Traces(float(d["fps"]), d["rgb"], list(d["regions"]), d["face_found"], d["pose"])

    fps = video_fps(path)
    rows, found, poses = [], [], []
    with FaceTracker(video_mode=True) as tracker:
        for i, frame in enumerate(iter_frames(path, max_frames, start_frame)):
            lm = tracker.landmarks(frame, timestamp_ms=i * 1000.0 / fps)
            found.append(lm is not None)
            if lm is None:
                rows.append(np.full((len(regions), 3), np.nan))
                poses.append(np.full(5, np.nan))
            else:
                rows.append(np.stack([region_mean_rgb(frame, lm, r) for r in regions]))
                poses.append(head_pose(lm))
            if verbose and i % 300 == 0:
                print(f"  {path.name}: frame {i}", flush=True)
    if not rows:
        raise ValueError(f"No frames read from {path}")
    rgb = _fill_gaps(_remove_spikes(np.asarray(rows, dtype=np.float64)))
    found = np.asarray(found)
    pose = _fill_gaps(np.asarray(poses, dtype=np.float64))
    tr = Traces(fps, rgb, regions, found, pose)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, fps=fps, rgb=rgb, regions=np.array(regions), face_found=found, pose=pose)
    return tr
