"""Step 4 - check the heart-rate reader against a real pulse (UBFC-rPPG).

    python eval_ubfc.py data/ubfc --windows 10 30

Expects subject folders containing vid.avi plus either ground_truth.txt (DATASET_2) or
gtdump.xmp (DATASET_1). For every window of video, the "true" BPM is read from the
recorded finger pulse (PPG) the same way as from video (strongest frequency x 60), and
compared with green / CHROM / POS. Score = mean absolute error (MAE) in BPM.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pulse.extract import extract_traces
from pulse.plots import INK2, style
from pulse.roi import DEFAULT_REGIONS
from pulse.signals import HR_BAND, METHODS, bandpass, detrend, estimate_bpm, pulse_wave


def load_ground_truth(folder):
    """Return (times_s, ppg) of the finger pulse sensor."""
    folder = Path(folder)
    gt2 = folder / "ground_truth.txt"
    gt1 = folder / "gtdump.xmp"
    if gt2.exists():
        rows = [np.array(line.split(), dtype=float) for line in gt2.read_text().strip().splitlines()]
        ppg, times = rows[0], rows[2]
    elif gt1.exists():
        d = np.loadtxt(gt1, delimiter=",")
        times, ppg = d[:, 0] / 1000.0, d[:, 3]
    else:
        raise FileNotFoundError(f"No ground_truth.txt or gtdump.xmp in {folder}")
    return times - times[0], ppg


def find_subjects(root):
    vids = sorted(Path(root).rglob("vid.avi"))
    return [v for v in vids if (v.parent / "ground_truth.txt").exists() or (v.parent / "gtdump.xmp").exists()]


def evaluate_subject(video, windows, step_s, band):
    tr = extract_traces(video, verbose=False)
    fs = tr.fps
    t_vid = np.arange(tr.n_frames) / fs
    gt_t, gt_ppg = load_ground_truth(video.parent)
    gt = np.interp(t_vid, gt_t, gt_ppg)                     # PPG resampled to video frames
    gt = bandpass(detrend(gt), fs, *band)
    idx = [tr.regions.index(r) for r in DEFAULT_REGIONS]
    skin = tr.rgb[:, idx, :].mean(axis=1)
    waves = {m: pulse_wave(skin, fs, m, band) for m in METHODS}
    rows = []
    for win in windows:
        l, st = int(win * fs), int(step_s * fs)
        for s in range(0, tr.n_frames - l + 1, st):
            sl = slice(s, s + l)
            row = {"subject": video.parent.name, "window_s": win, "start_s": s / fs,
                   "true_bpm": estimate_bpm(gt[sl], fs, band),
                   "face_found": tr.face_found[sl].mean()}
            for m in METHODS:
                row[m] = estimate_bpm(waves[m][sl], fs, band)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder containing the UBFC subject folders")
    ap.add_argument("--windows", type=float, nargs="+", default=[10, 30], help="window lengths in seconds")
    ap.add_argument("--step", type=float, default=5, help="seconds between windows")
    ap.add_argument("--limit", type=int, default=None, help="only the first N subjects")
    args = ap.parse_args()

    subjects = find_subjects(args.root)[: args.limit]
    if not subjects:
        raise SystemExit(f"No UBFC subjects (vid.avi + ground truth) found under {args.root}")
    print(f"{len(subjects)} subjects. Landmark extraction is cached, so re-runs are fast.")
    rows = []
    for k, v in enumerate(subjects, 1):
        print(f"[{k}/{len(subjects)}] {v.parent.name}", flush=True)
        try:
            rows += evaluate_subject(v, args.windows, args.step, HR_BAND)
        except Exception as e:  # keep going on one bad file
            print(f"   skipped: {e}")
    df = pd.DataFrame(rows)
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "ubfc_windows.csv", index=False)

    print("\nMean absolute error (BPM), averaged over subjects - lower is better")
    summary = []
    for win, g in df.groupby("window_s"):
        for m in METHODS:
            err = (g[m] - g["true_bpm"]).abs()
            per_subj = err.groupby(g["subject"]).mean()
            summary.append({"window_s": win, "method": m,
                            "MAE": per_subj.mean(),
                            "RMSE": np.sqrt(((g[m] - g["true_bpm"]) ** 2).mean()),
                            "within_5bpm_%": 100 * (err <= 5).mean(),
                            "pearson_r": np.corrcoef(g[m], g["true_bpm"])[0, 1]})
    s = pd.DataFrame(summary)
    print(s.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    s.to_csv(out / "ubfc_summary.csv", index=False)

    plt = style()
    wins = sorted(df["window_s"].unique())
    fig, axes = plt.subplots(1, len(METHODS), figsize=(11, 3.8), sharex=True, sharey=True)
    w = wins[-1]
    g = df[df["window_s"] == w]
    lim = [40, 160]
    for ax, m in zip(axes, METHODS):
        ax.plot(lim, lim, color=INK2, lw=1, ls="--")
        ax.scatter(g["true_bpm"], g[m], s=12, alpha=0.6, edgecolors="none")
        mae = s[(s.window_s == w) & (s.method == m)]["MAE"].iloc[0]
        ax.set_title(f"{m}: MAE {mae:.1f} BPM")
        ax.set_xlabel("true BPM (finger sensor)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    axes[0].set_ylabel("BPM from video")
    fig.suptitle(f"UBFC-rPPG, {w:.0f}-second windows - points on the dashed line are perfect", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "ubfc_scatter.png")
    print(f"\nSaved {(out / 'ubfc_summary.csv').resolve()}, ubfc_windows.csv and ubfc_scatter.png")


if __name__ == "__main__":
    main()
