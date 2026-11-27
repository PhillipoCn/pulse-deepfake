"""Synthetic-data tests: a known pulse goes in, the same pulse must come out."""
import cv2
import numpy as np
import pytest
from skimage import data

from pulse.extract import extract_traces
from pulse.features import FEATURE_NAMES, video_features
from pulse.extract import Traces
from pulse.signals import estimate_bpm, pulse_wave

FS = 30.0
PPG_RGB = np.array([0.33, 0.77, 0.53])  # relative pulse strength per channel in real skin


def synth_pulse(n, fs, bpm, rng, jitter=0.0):
    t = np.arange(n) / fs
    f = bpm / 60 + jitter * np.cumsum(rng.normal(0, 0.002, n))
    phase = 2 * np.pi * np.cumsum(np.full(n, 1.0) * f) / fs
    return np.sin(phase) + 0.3 * np.sin(2 * phase + 0.5)


@pytest.mark.parametrize("bpm", [55, 72, 110])
@pytest.mark.parametrize("method", ["green", "chrom", "pos"])
def test_bpm_from_trace(bpm, method):
    rng = np.random.default_rng(0)
    n = int(20 * FS)
    p = synth_pulse(n, FS, bpm, rng)
    base = np.array([180.0, 140.0, 120.0])
    drift = np.linspace(0, 3, n)[:, None]                       # slow lighting drift
    rgb = base + drift - 0.8 * p[:, None] * PPG_RGB + rng.normal(0, 0.5, (n, 3))
    est = estimate_bpm(pulse_wave(rgb, FS, method), FS)
    assert abs(est - bpm) < 3, (method, bpm, est)


def _write_video(path, n, bpm, rng):
    img = data.astronaut().astype(np.float32)
    p = synth_pulse(n, FS, bpm, rng)
    skin = (img[..., 0] > 95) & (img[..., 0] > img[..., 2] + 15) & (img[..., 0] > img[..., 1])
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FS, (img.shape[1], img.shape[0]))
    assert w.isOpened()
    for i in range(n):
        f = img.copy()
        f[skin] -= 1.5 * p[i] * PPG_RGB
        f += rng.normal(0, 1.0, f.shape)
        w.write(cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    w.release()


def test_end_to_end_video(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "synthetic_72bpm.avi"
    _write_video(path, int(12 * FS), 72, rng)
    tr = extract_traces(path, use_cache=False)
    assert tr.face_found.mean() > 0.95
    assert abs(tr.fps - FS) < 0.01
    face = tr.rgb.mean(axis=1)
    est = estimate_bpm(pulse_wave(face, tr.fps, "pos"), tr.fps)
    assert abs(est - 72) < 3, est


def _traces(coherent, rng, n=int(20 * FS), r=5):
    base = np.array([180.0, 140.0, 120.0])
    if coherent:
        p = synth_pulse(n, FS, 75, rng, jitter=1)
        pulses = [p * rng.uniform(0.6, 1.2) for _ in range(r)]
    else:
        pulses = [synth_pulse(n, FS, rng.uniform(50, 120), rng, jitter=1) * 0.5 for _ in range(r)]
    rgb = np.stack([base - pk[:, None] * PPG_RGB + rng.normal(0, 0.6, (n, 3)) for pk in pulses], axis=1)
    return Traces(FS, rgb, [f"r{k}" for k in range(r)], np.ones(n, bool))


def test_features_separate_coherent_from_incoherent():
    rng = np.random.default_rng(2)
    real = video_features(_traces(True, rng))
    fake = video_features(_traces(False, rng))
    assert real.shape[1] == len(FEATURE_NAMES)
    i = FEATURE_NAMES.index
    assert real[:, i("coh_mean")].mean() > fake[:, i("coh_mean")].mean()
    assert real[:, i("corr_mean")].mean() > fake[:, i("corr_mean")].mean()
    assert real[:, i("bpm_std")].mean() < fake[:, i("bpm_std")].mean()


def _noisy_traces(amp, seed, n=900, move_at=None):
    rng = np.random.default_rng(seed)
    p = synth_pulse(n, FS, 87, rng, jitter=1)
    pose = np.c_[np.full(n, 500.0), np.full(n, 800.0), np.full(n, 400.0), np.zeros(n), np.zeros(n)]
    pose[:, :2] += rng.normal(0, 0.5, (n, 2))
    rgb = np.stack([np.array([120, 100, 80.0]) - amp * p[:, None] * PPG_RGB
                    + np.cumsum(rng.normal(0, 0.05, (n, 3)), 0) + rng.normal(0, 0.3, (n, 3)) for _ in range(4)], 1)
    if move_at is not None:                       # a head turn that darkens the skin a lot
        sl = slice(move_at, move_at + 150)
        turn = np.sin(np.linspace(0, np.pi, 150))
        pose[sl, 0] += 200 * turn
        pose[sl, 3] += 0.3 * turn
        rgb[sl] -= 20 * turn[:, None, None]
    return Traces(FS, rgb, ["forehead_right", "forehead_left", "cheek_left", "cheek_right"], np.ones(n, bool), pose)


def test_robust_tracks_weak_pulse_and_skips_head_turn():
    from pulse.robust import robust_bpm
    r = robust_bpm(_noisy_traces(0.3, 1, move_at=400), nfft=1024)
    assert any(w.status == "motion" for w in r.windows)
    assert abs(r.track_bpm - 87) <= 4, r.track_bpm
    assert r.track_strength >= 2.3


def test_robust_noise_is_not_called_reliable():
    from pulse.robust import robust_bpm
    for seed in range(5):
        assert robust_bpm(_noisy_traces(0.0, seed), nfft=1024).track_strength < 3


def test_detail_features_see_frame_flicker():
    """Per-frame flicker (what face swaps add) must raise the flicker/high-band features."""
    from pulse.detail import DETAIL_NAMES, VIDEO_FEATURE_NAMES, detail_features, video_vector
    from pulse.roi import DETECTOR_REGIONS
    rng = np.random.default_rng(3)
    n = int(20 * FS)
    p = synth_pulse(n, FS, 72, rng)
    pose = np.c_[np.full(n, 500.0), np.full(n, 800.0), np.full(n, 400.0), np.zeros(n), np.zeros(n)]
    base = np.stack([np.array([120, 100, 80.0]) - 0.5 * p[:, None] * PPG_RGB
                     + rng.normal(0, 0.05, (n, 3)) for _ in DETECTOR_REGIONS], 1)
    shaky = base + rng.normal(0, 0.6, base.shape)          # independent frame-to-frame jitter
    clean = detail_features(Traces(FS, base, DETECTOR_REGIONS, np.ones(n, bool), pose))
    noisy = detail_features(Traces(FS, shaky, DETECTOR_REGIONS, np.ones(n, bool), pose))
    assert clean.shape == (len(DETAIL_NAMES),) and np.isfinite(clean).all()
    k = [i for i, nm in enumerate(DETAIL_NAMES) if nm.startswith("flicker_")]
    assert (noisy[k] > clean[k]).all()
    vec, _ = video_vector(Traces(FS, base, DETECTOR_REGIONS, np.ones(n, bool), pose))
    assert vec.shape == (len(VIDEO_FEATURE_NAMES),)
