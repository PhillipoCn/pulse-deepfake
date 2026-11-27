"""Step 5 - compare the patches. On a real face they rise and fall together.

Each window of video becomes one fixed-length vector of numbers describing how well the
skin patches agree with each other. These vectors are what the classifier (step 6) sees.
"""
from itertools import combinations

import numpy as np
from scipy import signal

from .signals import WIDE_BAND, bandpass, detrend, estimate_bpm, pos, snr_db, green

FEATURE_NAMES = [
    "coh_mean", "coh_min", "coh_std",        # POS coherence between patch pairs, 0.7-4 Hz
    "gcoh_mean", "gcoh_min",                 # same on the plain green channel
    "corr_mean", "corr_min",                 # correlation of filtered pulse waves
    "lag_mean",                              # mean |time lag| between patches (s)
    "bpm_std", "bpm_range",                  # do patches report the same heart rate?
    "snr_mean", "snr_min", "snr_face",       # how clean the pulse is
    "hf_ratio",                              # power 4-14 Hz vs 0.7-4 Hz (FakeCatcher's extra band)
]


def _pairwise_coherence(sigs, fs, band, nperseg):
    vals = []
    for i, j in combinations(range(sigs.shape[1]), 2):
        f, c = signal.coherence(sigs[:, i], sigs[:, j], fs=fs, nperseg=nperseg)
        m = (f >= band[0]) & (f <= band[1])
        vals.append(c[m].mean())
    return np.array(vals)


def _max_lag(a, b, fs, max_s=0.3):
    k = int(max_s * fs)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    xc = signal.correlate(a, b, mode="full")
    lags = signal.correlation_lags(len(a), len(b), mode="full")
    m = np.abs(lags) <= k
    return abs(lags[m][np.argmax(xc[m])]) / fs


def prepare_signals(rgb, fs):
    """Whole-video per-patch signals. rgb: (T, R, 3). Returns dict of (T, R) arrays."""
    t, r, _ = rgb.shape
    hi = min(14.0, 0.45 * fs)
    pos_raw = np.stack([pos(rgb[:, k], fs) for k in range(r)], axis=1)
    grn_raw = np.stack([green(rgb[:, k], fs) for k in range(r)], axis=1)
    return {
        "pos_wide": bandpass(detrend(pos_raw), fs, 0.7, hi),
        "pos_pulse": bandpass(detrend(pos_raw), fs, *WIDE_BAND),
        "green_wide": bandpass(detrend(grn_raw), fs, 0.7, hi),
        "face_rgb": rgb.mean(axis=1),
    }


def window_features(sig, fs, start, length):
    sl = slice(start, start + length)
    pw, pp, gw = sig["pos_wide"][sl], sig["pos_pulse"][sl], sig["green_wide"][sl]
    band = WIDE_BAND
    nperseg = min(64, length // 2)

    coh = _pairwise_coherence(pw, fs, band, nperseg)
    gcoh = _pairwise_coherence(gw, fs, band, nperseg)
    pairs = list(combinations(range(pp.shape[1]), 2))
    corr = np.array([np.corrcoef(pp[:, i], pp[:, j])[0, 1] for i, j in pairs])
    lags = np.array([_max_lag(pp[:, i], pp[:, j], fs) for i, j in pairs])

    bpms = np.array([estimate_bpm(pp[:, k], fs, band) for k in range(pp.shape[1])])
    snrs = np.array([snr_db(pp[:, k], fs, b, band) for k, b in enumerate(bpms)])

    face = bandpass(detrend(pos(sig["face_rgb"][sl], fs)), fs, *band)
    face_bpm = estimate_bpm(face, fs, band)

    f, pxx = signal.welch(gw, fs=fs, nperseg=nperseg, axis=0)
    lo = pxx[(f >= band[0]) & (f <= band[1])].sum(axis=0)
    hi_ = pxx[(f > band[1]) & (f <= min(14.0, 0.45 * fs))].sum(axis=0)

    return np.array([
        coh.mean(), coh.min(), coh.std(),
        gcoh.mean(), gcoh.min(),
        np.nanmean(corr), np.nanmin(corr),
        lags.mean(),
        bpms.std(), np.ptp(bpms),
        snrs.mean(), snrs.min(), snr_db(face, fs, face_bpm, band),
        np.log10((hi_ + 1e-20) / (lo + 1e-20)).mean(),
    ])


def video_features(traces, win_s=6.0, step_s=3.0):
    """All windows of one video -> (n_windows, n_features)."""
    fs = traces.fps
    length, step = int(round(win_s * fs)), max(int(round(step_s * fs)), 1)
    if traces.n_frames < length:
        raise ValueError(f"Video too short: {traces.duration:.1f}s < {win_s}s window")
    sig = prepare_signals(traces.rgb, fs)
    rows = [window_features(sig, fs, s, length) for s in range(0, traces.n_frames - length + 1, step)]
    x = np.asarray(rows)
    return x[np.isfinite(x).all(axis=1)]
