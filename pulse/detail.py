"""Whole-video detail features: keep every patch and colour channel separate.

features.py summarises how well the patches agree (FakeCatcher's idea). This module keeps
the full "PPG map" detail instead of averaging it away - per patch and per R/G/B channel:
  - spectral shape: share of power in 0.7-4, 4-7, 7-10 and 10-14 Hz
  - flicker: frame-to-frame change relative to the slow signal (face swaps re-render the
    face every frame, which adds jitter that a real face doesn't have)
plus cross-patch high-frequency correlation, head-pose jitter and red-green stability.
On FaceForensics++ c23 these lift video AUC from 0.78 to ~0.93 (README).
"""
import numpy as np
from scipy import signal

from .features import FEATURE_NAMES, video_features
from .roi import DETECTOR_REGIONS

BANDS = [(0.7, 4), (4, 7), (7, 10), (10, 14)]
CHANNELS = "RGB"
_PC = [f"{r}_{c}" for r in DETECTOR_REGIONS for c in CHANNELS]
_PAIRS = len(DETECTOR_REGIONS) * (len(DETECTOR_REGIONS) - 1) // 2

DETAIL_NAMES = (
    [f"band{lo}-{hi}_{pc}" for lo, hi in BANDS for pc in _PC]
    + [f"flicker_{pc}" for pc in _PC]
    + ["hf_xcorr"]
    + ["jitter_x", "jitter_y", "jitter_size", "jitter_yaw"]
    + [f"rg_std_{r}" for r in DETECTOR_REGIONS]
)
VIDEO_FEATURE_NAMES = ([f"{n}_mean" for n in FEATURE_NAMES] + [f"{n}_std" for n in FEATURE_NAMES]
                       + DETAIL_NAMES)


def _log(x):
    return np.log10(np.abs(x) + 1e-12)


def detail_features(traces):
    """One vector per video, length len(DETAIL_NAMES)."""
    fs, rgb, pose = traces.fps, traces.rgb, traces.pose
    if list(traces.regions) != DETECTOR_REGIONS:
        raise ValueError("traces must use DETECTOR_REGIONS")
    c = signal.detrend(rgb / rgb.mean(axis=0) - 1.0, axis=0)      # (T, R, 3) relative colour change
    top = min(14.0, 0.45 * fs)

    f, pxx = signal.welch(c, fs=fs, nperseg=min(128, len(c) // 2), axis=0)
    total = pxx[(f >= 0.7) & (f <= top)].sum(axis=0) + 1e-20
    bands = [_log(pxx[(f >= lo) & (f < min(hi, top))].sum(axis=0) / total) for lo, hi in BANDS]

    flicker = _log(np.diff(c, axis=0).std(axis=0) / (c.std(axis=0) + 1e-12))

    sos = signal.butter(3, min(4.0, 0.4 * fs) / (fs / 2), "high", output="sos")
    hf = signal.sosfiltfilt(sos, c[:, :, 1], axis=0)
    cc = np.corrcoef(hf.T)
    hf_xcorr = np.nanmean(cc[np.triu_indices(cc.shape[0], 1)])

    width = np.nanmedian(pose[:, 2])
    jitter = list(_log(np.abs(np.diff(pose[:, :3], n=2, axis=0) / width).mean(axis=0)))
    jitter.append(_log(np.abs(np.diff(pose[:, 3], n=2)).mean()))

    rg = _log((c[:, :, 0] - c[:, :, 1]).std(axis=0))

    v = np.concatenate([np.ravel(b) for b in bands] + [flicker.ravel(), [hf_xcorr], jitter, rg])
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def video_vector(traces, win_s=6.0, step_s=3.0):
    """Window features (mean + std over windows) + detail features -> one row per video."""
    w = video_features(traces, win_s, step_s)
    if len(w) == 0:
        raise ValueError("no valid feature windows")
    return np.concatenate([w.mean(axis=0), w.std(axis=0), detail_features(traces)]), w
