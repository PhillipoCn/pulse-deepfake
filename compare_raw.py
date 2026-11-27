"""Step 4 of the accuracy plan - does compression hide the pulse?

    python build_ffpp.py data/ffpp --compression raw --methods Deepfakes FaceSwap Face2Face \
        NeuralTextures FaceShifter --limit 100 --out outputs/ffpp_raw_features.npz
    python compare_raw.py

Uses the same people in raw (lossless) and c23 (YouTube-like) form:
  A. pulse quality and pulse-only detection, raw vs c23, same videos
  B. models trained on c23 without these people, tested on them at c23 and at raw
     (a change of compression, like going from YouTube to a phone camera)
"""
import warnings

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pulse.features import FEATURE_NAMES

warnings.filterwarnings("ignore")
PULSE = [i for i, n in enumerate(FEATURE_NAMES) if n != "hf_ratio"]


def per_video(d):
    vid = d["video"]
    first = np.array([np.flatnonzero(vid == v)[0] for v in range(len(d["stems"]))])
    keys = [f"{c}/{s}" for c, s in zip(d["category"][first], d["stems"])]
    win_mean = np.array([d["X"][vid == v].mean(0) for v in range(len(d["stems"]))])
    return keys, d["y"][first], d["groups"][first], d["category"][first], win_mean, d["V"]


def pulse_auc(X, y, g):
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.3))
    p = cross_val_predict(m, X, y, groups=g, cv=GroupKFold(5), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def show(name, y, p, cat, cut):
    ok = np.isfinite(p)
    y, p, cat = y[ok], p[ok], cat[ok]
    pred = (p > cut).astype(int)
    per = "  ".join(f"{c} {(pred == y)[cat == c].mean():.2f}" for c in np.unique(cat))
    print(f"  {name:28} AUC {roc_auc_score(y, p):.3f}  balanced {balanced_accuracy_score(y, pred):.3f}\n"
          f"  {'':28} {per}")


def main():
    raw = per_video(np.load("outputs/ffpp_raw_features.npz"))
    c23 = per_video(np.load("outputs/ffpp_features.npz"))
    both = sorted(set(raw[0]) & set(c23[0]))
    ri = np.array([raw[0].index(k) for k in both]); ci = np.array([c23[0].index(k) for k in both])
    y, g, cat = raw[1][ri], raw[2][ri], raw[3][ri]
    print(f"{len(both)} videos present in both raw and c23, {len(set(g))} identity groups")
    print("  " + "  ".join(f"{c} {(cat == c).sum()}" for c in np.unique(cat)))

    print("\nA. Is the pulse clearer without compression? (median over videos)")
    print(f"  {'':12}{'raw':>8}{'c23':>8}   (real videos only)")
    for n in ["snr_mean", "snr_face", "coh_mean", "corr_mean", "bpm_std", "hf_ratio"]:
        k = FEATURE_NAMES.index(n)
        r = raw[4][ri][y == 0, k]; c = c23[4][ci][y == 0, k]
        print(f"  {n:12}{np.median(r):8.2f}{np.median(c):8.2f}")
    print("\n  Detection from the 13 pulse-agreement features only (5-fold, unseen people), AUC:")
    print(f"    raw {pulse_auc(raw[4][ri][:, PULSE], y, g):.3f}   c23 {pulse_auc(c23[4][ci][:, PULSE], y, g):.3f}"
          "   (0.5 = coin flip)")

    # B: retrain on c23 WITHOUT anyone who appears in the raw subset, so the test is on unseen people
    ids_in_raw = {i for k in both for i in k.split("/")[1].split("_")}
    d23 = np.load("outputs/ffpp_features.npz")
    train_rows = np.array([not (set(k.split("/")[1].split("_")) & ids_in_raw) for k in c23[0]])
    print(f"\nB. Trained on c23 without these people ({train_rows.sum()} videos), tested on them")
    from train_detector import make_models
    boost = make_models("video")["boost"].fit(c23[5][train_rows], c23[1][train_rows])
    pb = boost.predict_proba(c23[5][train_rows])[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    cut = grid[np.argmax([balanced_accuracy_score(c23[1][train_rows], pb > t) for t in grid])]
    for name, V in [("c23", c23[5][ci]), ("raw", raw[5][ri])]:
        show(f"features+boost on {name}", y, boost.predict_proba(V)[:, 1], cat, cut)

    import torch
    from pulse.ppgmap import CROP, crops, ppg_map
    from train_cnn import load_maps, train
    maps = load_maps(d23, "data/ffpp", 8)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("  training the CNN on c23 without these people (~12 min)...", flush=True)
    net = train(maps, c23[1], np.flatnonzero(train_rows), device, 12)
    net = net.cpu().eval()
    for comp in ["c23", "raw"]:
        show(f"CNN on {comp}", y, cnn_on(net, both, comp), cat, 0.5)


def cnn_on(net, keys, compression):
    import torch
    from pulse.extract import extract_traces
    from pulse.ppgmap import CROP, crops, ppg_map
    from pulse.video import video_fps
    from train_cnn import video_path
    out = []
    for k in keys:
        cat, stem = k.split("/")
        p = str(video_path("data/ffpp", stem, cat, compression))
        tr = extract_traces(p, max_frames=int(20 * video_fps(p)))
        with torch.no_grad():
            out.append(torch.sigmoid(net(torch.tensor(crops(ppg_map(tr), step=64)))).mean().item()
                       if tr.n_frames >= CROP else np.nan)
    return np.array(out)


if __name__ == "__main__":
    main()
