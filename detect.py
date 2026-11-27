"""Step 8 - is this video real or fake?

    python detect.py some_video.mp4

Reads the pulse, compares the skin patches, and lets every window vote using the model
trained by train_detector.py. A learning project, not a forensic tool: see the caveats.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

from pulse.detail import VIDEO_FEATURE_NAMES, video_vector
from pulse.extract import extract_traces
from pulse.features import FEATURE_NAMES, video_features
from pulse.ppgmap import CROP, crops, make_net, ppg_map
from pulse.roi import DEFAULT_REGIONS
from pulse.signals import estimate_bpm, pulse_wave, snr_db
from pulse.video import video_fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--model", default="models/detector.joblib")
    ap.add_argument("--cnn", default="models/ppg_cnn.pt", help="PPG-map CNN (train_cnn.py); used for the verdict if present")
    ap.add_argument("--seconds", type=float, default=None, help="default: same length the model was trained on")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"No model at {args.model}. Run build_ffpp.py then train_detector.py first.")
    bundle = joblib.load(args.model)
    expected = VIDEO_FEATURE_NAMES if bundle.get("level", "window") == "video" else FEATURE_NAMES
    if list(bundle["features"]) != expected:
        raise SystemExit("Model was trained with different features - retrain it.")
    seconds = args.seconds or bundle["seconds"]
    fps = video_fps(args.video)
    tr = extract_traces(args.video, max_frames=int(seconds * fps))
    print(f"\n{Path(args.video).name}: {tr.duration:.1f} s analysed at {tr.fps:.1f} fps, "
          f"face found in {100 * tr.face_found.mean():.0f}% of frames")
    if tr.face_found.mean() < 0.3:
        raise SystemExit("No verdict: a face was found in too few frames to read a pulse.")
    if tr.duration < bundle["window"]:
        raise SystemExit(f"No verdict: the clip is shorter than one {bundle['window']:.0f} s window.")
    if tr.face_found.mean() < 0.8:
        print("WARNING: face often lost - result unreliable.")

    idx = [tr.regions.index(r) for r in DEFAULT_REGIONS]
    wave = pulse_wave(tr.rgb[:, idx, :].mean(axis=1), tr.fps, "pos")
    bpm = estimate_bpm(wave, tr.fps)
    snr = snr_db(wave, tr.fps, bpm)
    print(f"Heart rate read from the face: {bpm:.0f} BPM (signal quality {snr:.1f} dB)")

    if bundle.get("level", "window") == "video":
        vec, _ = video_vector(tr, bundle["window"], bundle["step"])
        pf = float(bundle["model"].predict_proba(vec[None])[0, 1])
        cut = bundle.get("threshold", 0.5)
        fake = pf > cut
        print(f"Whole-video model: P(fake) = {pf:.2f} (fake above {cut:.2f})")
    else:
        X = video_features(tr, bundle["window"], bundle["step"])
        p = bundle["model"].predict_proba(X)[:, 1]
        votes = (p > 0.5).mean()
        fake = votes > 0.5 or (votes == 0.5 and p.mean() > 0.5)
        print(f"{len(p)} windows: {int((p > 0.5).sum())} voted FAKE, {int((p <= 0.5).sum())} voted REAL "
              f"(mean P(fake) = {p.mean():.2f})")
    name, auc = bundle["name"], bundle["cv"]["video_auc"]
    if Path(args.cnn).exists() and tr.n_frames >= CROP:
        import torch
        ck = torch.load(args.cnn, map_location="cpu")
        net = make_net(); net.load_state_dict(ck["state"]); net.eval()
        with torch.no_grad():
            pc = torch.sigmoid(net(torch.tensor(crops(ppg_map(tr), step=64)))).mean().item()
        cut = ck.get("threshold", 0.5)
        print(f"PPG-map CNN:       P(fake) = {pc:.2f} (fake above {cut:.2f})  <- used for the verdict")
        fake, name, auc = pc > cut, "ppg-map CNN", ck["cv_auc"]
    print(f"\nVerdict: {'FAKE' if fake else 'REAL'}  "
          f"(model '{name}', tested at AUC {auc:.2f} on unseen people)")
    print("\nCaveats: compression, movement and uneven light cause false alarms; good face-swaps can\n"
          "copy the source video's pulse (2025), so a pulse alone does not prove a video is real.")


if __name__ == "__main__":
    main()
