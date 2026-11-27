"""Step 3 - read the heart rate from one video.

    python heart_rate.py data/mine/clip.avi --true-bpm 68

Prints BPM for the green, CHROM and POS methods and saves a plot of the cleaned pulse
wave and its spectrum to outputs/<video>_heart_rate.png.
"""
import argparse
from pathlib import Path

import numpy as np

from pulse.extract import extract_traces
from pulse.plots import INK2, style
from pulse.robust import robust_bpm
from pulse.roi import DEFAULT_REGIONS
from pulse.signals import HR_BAND, METHODS, estimate_bpm, pulse_wave, snr_db, spectrum


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=None, help="only use the first N seconds")
    ap.add_argument("--start", type=float, default=0.0, help="skip the first N seconds")
    ap.add_argument("--true-bpm", type=float, help="your real pulse (oximeter or wrist count)")
    ap.add_argument("--band", type=float, nargs=2, default=HR_BAND, metavar=("LO_HZ", "HI_HZ"))
    args = ap.parse_args()

    from pulse.video import video_fps
    fps = video_fps(args.video)
    tr = extract_traces(args.video,
                        max_frames=int(args.seconds * fps) if args.seconds else None,
                        start_frame=int(args.start * fps), verbose=True)
    print(f"\n{Path(args.video).name}: {tr.n_frames} frames at {tr.fps:.2f} fps = {tr.duration:.1f} s, "
          f"face found in {100 * tr.face_found.mean():.0f}% of frames")
    print(f"Frequency resolution {tr.fps / tr.n_frames:.3f} Hz = {60 * tr.fps / tr.n_frames:.1f} BPM "
          f"(longer clips tell beats apart more finely)\n")

    band = tuple(args.band)
    # Average the forehead + cheek patches into one skin trace
    idx = [tr.regions.index(r) for r in DEFAULT_REGIONS]
    skin = tr.rgb[:, idx, :].mean(axis=1)

    results = {}
    print(f"{'method':8} {'BPM':>6} {'SNR dB':>7}" + ("   error" if args.true_bpm else ""))
    for m in METHODS:
        wave = pulse_wave(skin, tr.fps, m, band)
        bpm = estimate_bpm(wave, tr.fps, band)
        snr = snr_db(wave, tr.fps, bpm, band)
        results[m] = (wave, bpm)
        err = f"  {bpm - args.true_bpm:+6.1f}" if args.true_bpm else ""
        print(f"{m:8} {bpm:6.1f} {snr:7.1f}{err}")
    print("\nPer patch (POS):")
    patch_bpms = []
    for r in DEFAULT_REGIONS:
        w = pulse_wave(tr.region(r), tr.fps, "pos", band)
        b = estimate_bpm(w, tr.fps, band)
        patch_bpms.append(b)
        print(f"  {r:15} {b:6.1f} BPM  SNR {snr_db(w, tr.fps, b, band):5.1f} dB")

    # Is there actually a pulse? If not, the "strongest frequency" is a random pick in the band.
    pos_wave, pos_bpm = results["pos"]
    pos_snr = snr_db(pos_wave, tr.fps, pos_bpm, band)
    spread = np.ptp(patch_bpms)
    problems = []
    if pos_snr < 0:
        problems.append(f"SNR {pos_snr:.1f} dB < 0: noise is stronger than any pulse")
    if spread > 10:
        problems.append(f"patches disagree by {spread:.0f} BPM (a real pulse is the same everywhere)")
    if problems:
        print("\nRESULT: NO RELIABLE PULSE - do not trust the BPM above.\n  - " + "\n  - ".join(problems))
        print("  Usual causes: head movement, dim or uneven light, auto-exposure, glasses/beard over patches.")
    else:
        print(f"\nRESULT: {pos_bpm:.0f} BPM (POS), signal looks reliable.")

    # ---- movement-robust reader (pulse/robust.py)
    rob = robust_bpm(tr, band=band, nfft=1024)
    n_motion = sum(w.status == "motion" for w in rob.windows)
    print(f"\nMovement-robust reader: {len(rob.windows)} windows of 10 s, "
          f"{n_motion} skipped for head movement")
    if np.isnan(rob.track_bpm):
        print("RESULT (robust): NO RELIABLE PULSE - too few still windows.")
        level = "none"
    else:
        level = "reliable" if rob.track_strength >= 3 else "weak" if rob.track_strength >= 2.3 else "none"
        label = {"reliable": "reliable",
                 "weak": "WEAK - treat it as a guess",
                 "none": "NO RELIABLE PULSE - pure noise scores this high too"}[level]
        print(f"RESULT (robust): {rob.track_bpm:.0f} BPM, pulse strength {rob.track_strength:.1f}x noise -> {label}")
        print("  (calibration on synthetic data: pure noise 1.6-2.2x, weak real pulse ~3x, clean pulse 5x+)")
    if not np.isnan(rob.track_bpm) and abs(rob.track_bpm - pos_bpm) > 10:
        print(f"  CHECK: the simple reader ({pos_bpm:.0f}) and robust reader ({rob.track_bpm:.0f}) disagree by "
              f"{abs(rob.track_bpm - pos_bpm):.0f} BPM - at least one is wrong; compare with a real pulse.")
    # One answer. Tested on 30 UBFC people (clean video, strength >= 3.9x): plain POS is the
    # most accurate (1.2 BPM error). On weak home video (2.2-3.6x) the tracker did better.
    st = rob.track_strength
    if np.isnan(st) or st < 2.3:
        best, why = float("nan"), "no reliable pulse in this video"
    elif st >= 4:
        best, why = pos_bpm, f"clean signal ({st:.1f}x) -> simple POS reading"
    else:
        best, why = rob.track_bpm, f"weak signal ({st:.1f}x) -> tracker reading, treat as a guess"
    print(f"\nBEST ANSWER: {'none' if np.isnan(best) else f'{best:.0f} BPM'}  ({why})")
    if args.true_bpm and not np.isnan(best):
        print(f"  error vs true pulse: {best - args.true_bpm:+.1f} BPM")

    plt = style()
    fig, axes = plt.subplots(4, 1, figsize=(9, 10.5))
    t = np.arange(tr.n_frames) / tr.fps
    axes[0].plot(t, skin[:, 1], color="#1baf7a")
    axes[0].set_title("Raw: average green of forehead + cheeks")
    axes[0].set_ylabel("green (0-255)")
    w, b = results["pos"]
    axes[1].plot(t, w)
    axes[1].set_title(f"Cleaned pulse wave (POS, {band[0]}-{band[1]} Hz)")
    axes[1].set_xlabel("seconds")
    axes[1].set_yticklabels([])
    for k, m in enumerate(METHODS):
        f, p = spectrum(results[m][0], tr.fps, 4096)
        sel = (f >= 0.5) & (f <= 4.5)
        axes[2].plot(f[sel] * 60, p[sel] / p[sel].max(), label=f"{m}: {results[m][1]:.0f} BPM")
    if args.true_bpm:
        axes[2].axvline(args.true_bpm, color=INK2, ls="--", lw=1)
        axes[2].text(args.true_bpm + 2, 0.95, f"true {args.true_bpm:.0f}", color=INK2)
    axes[2].set_title("Spectrum - the tallest spike is the heart rate")
    axes[2].set_xlabel("beats per minute")
    axes[2].set_ylabel("relative power")
    axes[2].legend(loc="upper right")
    ax = axes[3]
    for status, colour, name in [("ok", "#1baf7a", "window ok"), ("weak", "#eda100", "window weak"),
                                 ("motion", "#eb6834", "skipped: head moving")]:
        ws = [w for w in rob.windows if w.status == status]
        if ws:
            ax.scatter([w.start_s + 5 for w in ws], [w.bpm for w in ws], s=28, color=colour, label=name, zorder=3)
    still = [w for w in rob.windows if w.status != "motion"]
    if rob.track:
        ax.plot([w.start_s + 5 for w in still], rob.track, color="#2a78d6", label="tracked heart rate")
    if args.true_bpm:
        ax.axhline(args.true_bpm, color=INK2, ls="--", lw=1)
        ax.text(0.01, args.true_bpm, f" true {args.true_bpm:.0f}", color=INK2, va="bottom",
                transform=ax.get_yaxis_transform())
    ax.set_title("Movement-robust reader: BPM per 10-second window")
    ax.set_xlabel("seconds (window centre)")
    ax.set_ylabel("BPM")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    out = Path("outputs") / f"{Path(args.video).stem}_heart_rate.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"\nPlot saved: {out.resolve()}")


if __name__ == "__main__":
    main()
