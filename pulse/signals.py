"""Step 3 - turn a colour trace into a pulse wave, then into beats per minute.

Three ways to get a pulse wave from a patch's (T, 3) RGB trace:
  green  - just the green channel (simplest; Verkruysse 2008)
  chrom  - de Haan & Jeanne 2013, chrominance combination of R, G, B
  pos    - Wang et al. 2017, "plane orthogonal to skin"
"""
import numpy as np
from scipy import signal

HR_BAND = (0.7, 3.0)      # 42-180 BPM: the doc's starting band
WIDE_BAND = (0.7, 4.0)    # 42-240 BPM


def detrend(x):
    """Remove slow drift (lighting changes, auto-exposure)."""
    return signal.detrend(x, axis=0)


def bandpass(x, fs, lo, hi, order=3):
    """Keep only frequencies between lo and hi Hz (zero-phase Butterworth)."""
    hi = min(hi, 0.45 * fs)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    padlen = min(3 * (2 * len(sos) + 1), len(x) - 1)
    return signal.sosfiltfilt(sos, x, axis=0, padlen=padlen)


# ---------------------------------------------------------------- pulse wave methods

def green(rgb, fs):
    g = rgb[:, 1]
    return detrend(g / np.mean(g) - 1.0)


def chrom(rgb, fs, band=WIDE_BAND):
    n = rgb / rgb.mean(axis=0) - 1.0
    r, g, b = n[:, 0], n[:, 1], n[:, 2]
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    xf, yf = bandpass(detrend(x), fs, *band), bandpass(detrend(y), fs, *band)
    alpha = np.std(xf) / (np.std(yf) + 1e-12)
    return xf - alpha * yf


def pos(rgb, fs, win_s=1.6):
    t = len(rgb)
    l = max(int(round(win_s * fs)), 2)
    h = np.zeros(t)
    proj = np.array([[0, 1, -1], [-2, 1, 1]], dtype=float)
    for start in range(0, t - l + 1):
        c = rgb[start:start + l]
        cn = c / (c.mean(axis=0) + 1e-12)
        s = cn @ proj.T                      # (l, 2)
        p = s[:, 0] + (np.std(s[:, 0]) / (np.std(s[:, 1]) + 1e-12)) * s[:, 1]
        h[start:start + l] += p - p.mean()
    return detrend(h)


METHODS = {"green": green, "chrom": chrom, "pos": pos}


def pulse_wave(rgb, fs, method="pos", band=HR_BAND):
    """Clean pulse wave: method -> detrend -> band-pass."""
    raw = METHODS[method](np.asarray(rgb, dtype=float), fs)
    return bandpass(detrend(raw), fs, *band)


# ---------------------------------------------------------------- reading the beats

def spectrum(x, fs, pad_to=None):
    """Power at each frequency (numpy rfft). Zero-padding gives a smoother curve but does
    NOT improve the true resolution, which is fs / len(x) Hz (e.g. 0.1 Hz for 10 s)."""
    x = np.asarray(x, dtype=float) * np.hanning(len(x))
    n = max(len(x), pad_to or 0)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(x, n=n)) ** 2
    return freqs, power


def estimate_bpm(x, fs, band=HR_BAND, pad_to=4096):
    """Strongest frequency inside the heart-rate band, times 60."""
    freqs, power = spectrum(x, fs, pad_to)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any():
        return float("nan")
    return float(freqs[m][np.argmax(power[m])] * 60.0)


def snr_db(x, fs, bpm, band=HR_BAND, pad_to=4096):
    """Signal-to-noise ratio: power near the heart rate (and its 2nd harmonic) vs the rest
    of the band, in dB (de Haan 2013 style). Higher = cleaner pulse."""
    freqs, power = spectrum(x, fs, pad_to)
    f0 = bpm / 60.0
    m = (freqs >= band[0]) & (freqs <= min(2 * band[1], fs / 2))
    near = (np.abs(freqs - f0) <= 0.1) | (np.abs(freqs - 2 * f0) <= 0.2)
    sig = power[m & near].sum()
    noise = power[m & ~near].sum()
    return float(10 * np.log10((sig + 1e-20) / (noise + 1e-20)))


def windowed_bpm(x, fs, win_s=10.0, step_s=1.0, band=HR_BAND):
    """BPM in sliding windows. Returns (window_centre_seconds, bpm)."""
    l, st = int(win_s * fs), max(int(step_s * fs), 1)
    centres, bpms = [], []
    for s in range(0, len(x) - l + 1, st):
        centres.append((s + l / 2) / fs)
        bpms.append(estimate_bpm(x[s:s + l], fs, band))
    return np.array(centres), np.array(bpms)
