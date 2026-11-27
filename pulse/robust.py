"""Movement-robust heart rate.

The simple reader averages all patches over the whole clip, so one head turn can swamp
the pulse. This one works window by window:
  1. skip windows where the head turns, moves fast, or is turned far enough to hide a cheek
  2. cancel what's left of the movement: remove the part of each colour trace that
     follows the head's position, size and angle (least-squares regression)
  3. let the patches vote: add their normalised spectra. A real pulse sits at the same
     frequency in every patch and adds up; noise peaks land in different places.
  4. answer = median over the good windows, or "no reliable pulse" if none are good.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .roi import DEFAULT_REGIONS
from .signals import HR_BAND, bandpass, detrend, pos


@dataclass
class WindowResult:
    start_s: float
    bpm: float
    snr_db: float          # combined-spectrum peak vs rest of band
    agreement: float       # share of patches whose own peak is near the combined one
    yaw_max: float
    speed: float           # mean head speed, face-widths per second
    status: str            # "ok", "motion", "weak"


@dataclass
class RobustResult:
    bpm: float             # nan if no reliable pulse
    windows: list = field(default_factory=list)
    track_bpm: float = float("nan")   # median of the continuous track (Viterbi)
    track: list = field(default_factory=list)
    track_strength: float = float("nan")  # mean power share on the track / noise share

    @property
    def good(self):
        return [w for w in self.windows if w.status == "ok"]


def head_motion(pose, fs):
    """Per-frame head speed in face-widths per second, and |yaw|."""
    width = np.nanmedian(pose[:, 2])
    step = np.hypot(np.diff(pose[:, 0]), np.diff(pose[:, 1])) / width
    scale = np.abs(np.diff(pose[:, 2])) / width
    speed = np.concatenate([[0], (step + scale) * fs])
    return speed, np.abs(pose[:, 3])


def remove_motion(rgb, pose):
    """Regress the window's colour trace on head pose and subtract the fit.
    rgb: (L, 3), pose: (L, 5). Returns a cleaned (L, 3) trace with the same mean."""
    mean = rgb.mean(axis=0)
    c = rgb / mean - 1.0
    p = pose - pose.mean(axis=0)
    p = p / (p.std(axis=0) + 1e-9)
    t = np.linspace(-1, 1, len(rgb))[:, None]
    a = np.hstack([np.ones((len(rgb), 1)), t, p])
    coef, *_ = np.linalg.lstsq(a, c, rcond=None)
    c_clean = c - a[:, 2:] @ coef[2:]       # keep intercept/trend, remove the motion part
    return mean * (1.0 + c_clean)


def _norm_spectrum(x, fs, band, nfft):
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nfft)) ** 2
    m = (f >= band[0]) & (f <= band[1])
    return f[m], p[m] / (p[m].sum() + 1e-20)


def robust_bpm(tr, regions=None, win_s=10.0, step_s=2.0, band=HR_BAND,
               max_yaw=0.3, max_yaw_change=0.1, max_speed=0.6, min_snr=-1.0, min_agreement=0.5,
               motion_gate=True, motion_regress=True, spectral_vote=True, nfft=4096):
    fs = tr.fps
    regions = regions or DEFAULT_REGIONS
    idx = [tr.regions.index(r) for r in regions]
    L, S = int(round(win_s * fs)), max(int(round(step_s * fs)), 1)
    have_pose = tr.pose is not None and np.isfinite(tr.pose).all()
    speed, yaw = head_motion(tr.pose, fs) if have_pose else (np.zeros(tr.n_frames),) * 2

    windows, still_spectra = [], []
    for s in range(0, tr.n_frames - L + 1, S):
        sl = slice(s, s + L)
        ym, sp = float(np.percentile(yaw[sl], 90)), float(np.mean(speed[sl]))
        # A steady head angle is fine; turning during the window, moving fast, or turning
        # so far that a cheek leaves the face is not.
        y_change = float(np.percentile(yaw[sl], 95) - np.percentile(yaw[sl], 5))
        moving = motion_gate and have_pose and (ym > max_yaw or y_change > max_yaw_change or sp > max_speed)

        spectra, peaks = [], []
        patch_rgb = [tr.rgb[sl, k, :] for k in idx]
        if motion_regress and have_pose:
            patch_rgb = [remove_motion(c, tr.pose[sl]) for c in patch_rgb]
        if not spectral_vote:                       # plain average of patches, like the simple reader
            patch_rgb = [np.mean(patch_rgb, axis=0)]
        for c in patch_rgb:
            w = bandpass(detrend(pos(c, fs)), fs, *band)
            f, p = _norm_spectrum(w, fs, band, nfft)
            spectra.append(p)
            peaks.append(f[np.argmax(p)])
        combo = np.mean(spectra, axis=0)
        if not moving:
            still_spectra.append(combo)
        f0 = f[np.argmax(combo)]
        near = np.abs(f - f0) <= 0.1
        snr = 10 * np.log10(combo[near].sum() / (combo[~near].sum() + 1e-20))
        agree = float(np.mean(np.abs(np.array(peaks) - f0) * 60 <= 6))
        if moving:
            status = "motion"
        elif snr < min_snr or (spectral_vote and agree < min_agreement):
            status = "weak"
        else:
            status = "ok"
        windows.append(WindowResult(s / fs, f0 * 60, float(snr), agree, ym, sp, status))

    good = [w.bpm for w in windows if w.status == "ok"]
    res = RobustResult(float(np.median(good)) if good else float("nan"), windows)
    if len(still_spectra) >= 3:
        path = viterbi_track(np.array(still_spectra), f * 60, step_s)
        res.track = list(f[path] * 60)
        res.track_bpm = float(np.median(res.track))
        # Share of power within +-3 BPM of the track, compared with what flat noise would give
        width = (np.abs(f * 60 - f[path[0]] * 60) <= 3).sum() / len(f)
        share = np.mean([sp[np.abs(f * 60 - f[i] * 60) <= 3].sum() for sp, i in zip(still_spectra, path)])
        res.track_strength = float(share / width)
    return res


def viterbi_track(spectra, bpm_axis, step_s, sigma_bpm_per_s=2.5):
    """Most likely slowly-changing heart-rate path through a sequence of spectra.
    Score = sum of log power on the path minus a Gaussian penalty on jumps
    (heart rate rarely changes more than ~5 BPM in 2 s while sitting still)."""
    logp = np.log(spectra + 1e-12)
    sigma = sigma_bpm_per_s * step_s
    d = bpm_axis[:, None] - bpm_axis[None, :]
    trans = -0.5 * (d / sigma) ** 2
    score = logp[0].copy()
    back = np.zeros(spectra.shape, int)
    for t in range(1, len(spectra)):
        cand = score[None, :] + trans          # [to, from]
        back[t] = np.argmax(cand, axis=1)
        score = cand[np.arange(len(bpm_axis)), back[t]] + logp[t]
    path = [int(np.argmax(score))]
    for t in range(len(spectra) - 1, 0, -1):
        path.append(back[t][path[-1]])
    return path[::-1]
