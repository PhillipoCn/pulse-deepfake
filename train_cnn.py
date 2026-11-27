"""Step 3 of the accuracy plan - a small CNN on PPG maps (FakeCatcher's method).

    python train_cnn.py                          # 5-fold test on unseen people, + ensemble with boost
    python train_cnn.py --holdout-method FaceSwap

Needs outputs/ffpp_features.npz (build_ffpp.py) - it reuses that file's video list, labels
and identity groups, and the cached face traces, so no video is decoded again.
Saves models/ppg_cnn.pt (trained on everything).
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
MAPS = ROOT / "outputs" / "ppg_maps.npz"


def video_path(data_root, stem, cat, compression="c23"):
    if cat == "real":
        return Path(data_root) / "original_sequences" / "youtube" / compression / "videos" / f"{stem}.mp4"
    return Path(data_root) / "manipulated_sequences" / cat / compression / "videos" / f"{stem}.mp4"


def _map_one(args):
    path, seconds = args
    os.environ.setdefault("GLOG_minloglevel", "2")
    from pulse.extract import extract_traces
    from pulse.ppgmap import ppg_map
    from pulse.video import video_fps
    tr = extract_traces(path, max_frames=int(seconds * video_fps(path)))
    return ppg_map(tr)


def load_maps(d, data_root, workers):
    stems = list(d["stems"])
    if MAPS.exists():
        m = np.load(MAPS, allow_pickle=False)
        if list(m["stems"]) == stems:
            big, lengths = m["maps"], m["lengths"]          # read once, then slice
            return [big[i, :, :lengths[i]] for i in range(len(stems))]
    vid, cat = d["video"], d["category"]
    vcat = [cat[np.flatnonzero(vid == v)[0]] for v in range(len(stems))]
    jobs = [(str(video_path(data_root, s, c)), float(d["seconds"])) for s, c in zip(stems, vcat)]
    print(f"Building PPG maps for {len(jobs)} videos from cached traces...")
    with ProcessPoolExecutor(workers) as ex:
        maps = list(ex.map(_map_one, jobs, chunksize=20))
    lengths = np.array([m.shape[1] for m in maps])
    big = np.zeros((len(maps), maps[0].shape[0], lengths.max()), np.float32)
    for i, m in enumerate(maps):
        big[i, :, :m.shape[1]] = m
    np.savez(MAPS, maps=big, lengths=lengths, stems=np.array(stems))
    return maps


def train(maps, y, idx, device, epochs, seed=0, crops_per_video=4, verbose=False):
    import torch
    from pulse.ppgmap import CROP, make_net
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    per_epoch = -(-len(idx) * crops_per_video // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * per_epoch)
    yt = y[idx]
    pos_w = torch.tensor(float((yt == 0).sum() / max((yt == 1).sum(), 1)), dtype=torch.float32, device=device)   # balance classes
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(epochs):
        net.train()
        xs, ys = [], []
        for i in idx:                                  # random crops = free augmentation
            m = maps[i]
            for s in rng.integers(0, m.shape[1] - CROP + 1, crops_per_video):
                xs.append(m[:, s:s + CROP]); ys.append(y[i])
        order = rng.permutation(len(xs))
        X = torch.tensor(np.stack(xs)[order], device=device)
        Y = torch.tensor(np.array(ys, np.float32)[order], device=device)
        total = 0.0
        for b in range(0, len(X), 256):
            opt.zero_grad()
            loss = lossf(net(X[b:b + 256]), Y[b:b + 256])
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(X[b:b + 256])
        if verbose:
            print(f"    epoch {ep + 1}/{epochs} loss {total / len(X):.3f}", flush=True)
    return net


def predict(net, maps, idx, device):
    import torch
    from pulse.ppgmap import crops
    net.eval()
    out = []
    with torch.no_grad():
        for i in idx:
            x = torch.tensor(crops(maps[i], step=64), device=device)
            out.append(torch.sigmoid(net(x)).mean().item())
    return np.array(out)


def boost_oof(V, y, groups, folds):
    from train_detector import make_models
    p = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(V, y, groups):
        p[te] = make_models("video")["boost"].fit(V[tr], y[tr]).predict_proba(V[te])[:, 1]
    return p


def summary(name, y, p, cat):
    pred = (p > 0.5).astype(int)
    per = "  ".join(f"{c} {((pred == y)[cat == c]).mean():.2f}" for c in np.unique(cat))
    print(f"{name:22} acc {np.mean(pred == y):.3f}  balanced {balanced_accuracy_score(y, pred):.3f}  "
          f"AUC {roc_auc_score(y, p):.3f}\n{'':22} {per}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="?", default="outputs/ffpp_features.npz")
    ap.add_argument("--data", default="data/ffpp")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--holdout-method", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--save", default="models/ppg_cnn.pt")
    args = ap.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    d = np.load(args.features, allow_pickle=False)
    maps = load_maps(d, args.data, args.workers)
    vid = d["video"]
    first = np.array([np.flatnonzero(vid == v)[0] for v in range(len(maps))])
    y, groups, cat = d["y"][first], d["groups"][first], d["category"][first]
    keep = np.array([m.shape[1] >= 128 for m in maps])
    print(f"{keep.sum()} videos ({(~keep).sum()} too short), device {device}")
    ids = np.flatnonzero(keep)

    if args.holdout_method:   # same split rule as train_detector.py, so the numbers compare
        test_groups = np.unique(groups[cat == args.holdout_method])
        test_groups = np.random.default_rng(0).choice(test_groups, size=max(1, len(test_groups) // 3), replace=False)
        te = ids[np.isin(groups[ids], test_groups) & ((cat[ids] == "real") | (cat[ids] == args.holdout_method))]
        tr = ids[~np.isin(groups[ids], test_groups) & (cat[ids] != args.holdout_method)]
        net = train(maps, y, tr, device, args.epochs)
        p = predict(net, maps, te, device)
        print(f"\nHold-out: trained without {args.holdout_method}, tested on it + real, unseen people")
        summary("cnn", y[te], p, cat[te])
        return

    p_cnn = np.zeros(len(ids))
    for k, (tr, te) in enumerate(GroupKFold(args.folds).split(ids, y[ids], groups[ids]), 1):
        net = train(maps, y, ids[tr], device, args.epochs)
        p_cnn[te] = predict(net, maps, ids[te], device)
        print(f"  fold {k}/{args.folds}: AUC {roc_auc_score(y[ids[te]], p_cnn[te]):.3f}", flush=True)

    print(f"\n{args.folds}-fold GroupKFold, video level, people never seen")
    summary("cnn on PPG maps", y[ids], p_cnn, cat[ids])
    if "V" in d.files:
        p_b = boost_oof(d["V"][ids], y[ids], groups[ids], args.folds)
        summary("boost (features)", y[ids], p_b, cat[ids])
        summary("average of both", y[ids], (p_cnn + p_b) / 2, cat[ids])

    net = train(maps, y, ids, device, args.epochs)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.05, 0.95, 91)
    bal = [balanced_accuracy_score(y[ids], (p_cnn > t).astype(int)) for t in grid]
    threshold = float(grid[int(np.argmax(bal))])
    print(f"Balanced cut-off: P(fake) > {threshold:.2f} -> balanced acc {max(bal):.3f}")
    np.savez(ROOT / "outputs" / "cnn_oof.npz", p=p_cnn, y=y[ids], cat=cat[ids], stems=d["stems"][ids])
    torch.save({"state": net.state_dict(), "seconds": float(d["seconds"]), "threshold": threshold,
                "cv_auc": float(roc_auc_score(y[ids], p_cnn))}, args.save)
    print(f"Saved {Path(args.save).resolve()}")


if __name__ == "__main__":
    main()
