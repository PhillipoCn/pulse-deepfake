"""PPG maps + a small CNN - FakeCatcher's own idea (step 3 of the accuracy plan).

Instead of hand-made summary numbers, keep the whole signal as a map:
rows = 5 patches x R/G/B relative colour change (15), their frame-to-frame change (15),
and each patch's POS pulse wave (5) -> 35 rows x time. A 1D CNN slides along time and
learns its own patterns. A video's score is the average over its 128-frame crops.
"""
import numpy as np
from scipy import signal

from .roi import DETECTOR_REGIONS
from .signals import bandpass, detrend, pos

CROP = 128                # frames per CNN input (FakeCatcher used 128-frame windows)
N_ROWS = 35


def ppg_map(traces):
    """(35, T) float32 map, each row standardised over the whole video."""
    if list(traces.regions) != DETECTOR_REGIONS:
        raise ValueError("traces must use DETECTOR_REGIONS")
    fs, rgb = traces.fps, traces.rgb
    c = signal.detrend(rgb / rgb.mean(axis=0) - 1.0, axis=0)                  # (T, 5, 3)
    rows = [c.reshape(len(c), -1)]
    d = np.diff(c, axis=0, prepend=c[:1])
    rows.append(d.reshape(len(d), -1))
    rows.append(np.stack([bandpass(detrend(pos(rgb[:, k], fs)), fs, 0.7, 4.0) for k in range(rgb.shape[1])], 1))
    m = np.concatenate(rows, axis=1).T                                         # (35, T)
    m = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)
    return np.nan_to_num(m, nan=0.0).astype(np.float32)


def crops(m, step=CROP):
    """All crops of CROP frames, stacked -> (n, 35, CROP)."""
    starts = range(0, m.shape[1] - CROP + 1, step)
    return np.stack([m[:, s:s + CROP] for s in starts])


def make_net():
    import torch.nn as nn

    def block(i, o, k, s):
        return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.BatchNorm1d(o), nn.ReLU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(block(N_ROWS, 64, 7, 1), block(64, 64, 5, 2), block(64, 128, 5, 2),
                                      block(128, 128, 3, 2), nn.Dropout(0.2))
            self.head = nn.Linear(256, 1)

        def forward(self, x):
            h = self.body(x)
            import torch
            return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1)).squeeze(1)

    return Net()
