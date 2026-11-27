"""Steps 6+7 - train the real/fake classifier and test it on people it has never seen.

    python train_detector.py outputs/ffpp_features.npz
    python train_detector.py outputs/ffpp_features.npz --holdout-method FaceSwap

Default (--level video): one row per video = window features (mean + std) plus the
per-patch, per-colour detail features (detail.py). --level window: the original
FakeCatcher-style setup, one row per 6 s window and a majority vote per video. GroupKFold keeps each identity group (a real video and every fake made
from it) on one side of the train/test line, so the score isn't flattered.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pulse.detail import VIDEO_FEATURE_NAMES
from pulse.features import FEATURE_NAMES
from pulse.plots import style


def make_models(level="window"):
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=5000, C=1.0 if level == "window" else 0.3)),
        "svm": make_pipeline(StandardScaler(), CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, class_weight="balanced"), ensemble=False)),
    }
    if level == "video":   # boosted trees: picks up which patch/channel/band matters
        models["boost"] = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            class_weight="balanced", random_state=0)
    return models


def video_votes(prob, video):
    """Per video: share of windows voting fake, and the true label is filled in by caller."""
    vids = np.unique(video)
    return vids, np.array([(prob[video == v] > 0.5).mean() for v in vids]), \
        np.array([prob[video == v].mean() for v in vids])


def report(name, y_win, prob_win, video, cat):
    vids, vote, meanp = video_votes(prob_win, video)
    y_vid = np.array([y_win[video == v][0] for v in vids])
    cat_vid = np.array([cat[video == v][0] for v in vids])
    pred_vid = np.where(vote == 0.5, meanp > 0.5, vote > 0.5).astype(int)  # ties -> mean prob
    res = {
        "model": name,
        "window_acc": ((prob_win > 0.5) == y_win).mean(),
        "video_acc": (pred_vid == y_vid).mean(),
        "video_bal_acc": balanced_accuracy_score(y_vid, pred_vid),
        "video_auc": roc_auc_score(y_vid, meanp) if len(set(y_vid)) == 2 else float("nan"),
        "chance": max(y_vid.mean(), 1 - y_vid.mean()),
    }
    per_cat = {c: (pred_vid[cat_vid == c] == y_vid[cat_vid == c]).mean() for c in np.unique(cat_vid)}
    return res, per_cat, (y_vid, meanp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="?", default="outputs/ffpp_features.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdout-method", default=None,
                    help="train without this fake method, test only on it (does it generalise?)")
    ap.add_argument("--level", choices=["video", "window"], default="video",
                    help="video: one row per video with detail features (default); window: original votes")
    ap.add_argument("--save", default="models/detector.joblib")
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=False)
    X, y, video, groups, cat = d["X"], d["y"], d["video"], d["groups"], d["category"]
    print(f"{len(X)} windows, {len(np.unique(video))} videos, {len(np.unique(groups))} identity groups")
    feature_names = FEATURE_NAMES
    if args.level == "video":
        if "V" not in d.files:
            raise SystemExit("No video-level features in this file - rerun build_ffpp.py, or use --level window.")
        first = np.array([np.flatnonzero(video == v)[0] for v in range(len(d["V"]))])
        y, groups, cat = y[first], groups[first], cat[first]
        X, video = d["V"], np.arange(len(first))
        feature_names = list(d["video_feature_names"])
        if feature_names != VIDEO_FEATURE_NAMES:
            raise SystemExit("Feature file is from an older version - rerun build_ffpp.py.")
        print(f"Video level: {X.shape[1]} features per video")
    for c in np.unique(cat):
        print(f"  {c:15} {len(np.unique(video[cat == c]))} videos")

    if args.holdout_method:
        others = set(cat) - {"real", args.holdout_method}
        if args.holdout_method not in set(cat) or not others:
            raise SystemExit("--holdout-method needs that method plus at least one other fake method in the "
                             "features file (build_ffpp.py --methods Deepfakes FaceSwap ...).")
        test_groups = np.unique(groups[cat == args.holdout_method])
        rng = np.random.default_rng(0)
        test_groups = rng.choice(test_groups, size=max(1, len(test_groups) // 3), replace=False)
        te = np.isin(groups, test_groups) & ((cat == "real") | (cat == args.holdout_method))
        tr = ~np.isin(groups, test_groups) & (cat != args.holdout_method)
        print(f"\nHold-out test: train on {sorted(set(cat[tr]) - {'real'})}, test on {args.holdout_method} "
              f"(+ real) from {len(test_groups)} unseen identity groups")
        for name, model in make_models(args.level).items():
            model.fit(X[tr], y[tr])
            res, per_cat, _ = report(name, y[te], model.predict_proba(X[te])[:, 1], video[te], cat[te])
            print(f"  {name:7} video acc {res['video_acc']:.3f}  balanced {res['video_bal_acc']:.3f}  "
                  f"AUC {res['video_auc']:.3f}  (chance {res['chance']:.3f})")
        return

    gkf = GroupKFold(n_splits=args.folds)
    results = {}
    for name, model in make_models(args.level).items():
        prob = np.zeros(len(X))
        for tr_idx, te_idx in gkf.split(X, y, groups):
            assert not set(groups[tr_idx]) & set(groups[te_idx])
            model.fit(X[tr_idx], y[tr_idx])
            prob[te_idx] = model.predict_proba(X[te_idx])[:, 1]
        results[name] = report(name, y, prob, video, cat)

    print(f"\n{args.folds}-fold GroupKFold - every score is on people the model never saw")
    print(f"{'model':8}{'row acc':>12}{'video acc':>11}{'balanced':>10}{'AUC':>7}{'chance':>8}")
    for name, (res, per_cat, _) in results.items():
        print(f"{name:8}{res['window_acc']:12.3f}{res['video_acc']:11.3f}{res['video_bal_acc']:10.3f}"
              f"{res['video_auc']:7.3f}{res['chance']:8.3f}")
    print("\nVideo accuracy by category:")
    for name, (_, per_cat, _) in results.items():
        print(f"  {name:7} " + "  ".join(f"{c} {v:.2f}" for c, v in per_cat.items()))
    print("\n'balanced' averages accuracy on real and on fake, so it isn't fooled by having more fakes.\n"
          "AUC 0.5 = coin flip, 1.0 = perfect. Above chance on unseen people is a good result.")

    best = max(results, key=lambda k: results[k][0]["video_auc"])
    # With many fake types per real video, 0.5 leans towards "fake". Pick the cut-off that
    # balances accuracy on real and fake (chosen on the held-out predictions above).
    y_vid, p_vid = results[best][2]
    grid = np.linspace(0.05, 0.95, 91)
    bal = [balanced_accuracy_score(y_vid, (p_vid > t).astype(int)) for t in grid]
    threshold = float(grid[int(np.argmax(bal))])
    pred = (p_vid > threshold).astype(int)
    cat_vid = np.array([cat[video == v][0] for v in np.unique(video)])
    print(f"\nBalanced cut-off for {best}: P(fake) > {threshold:.2f} -> balanced acc {max(bal):.3f}")
    print("  " + "  ".join(f"{c} {(pred == y_vid)[cat_vid == c].mean():.2f}" for c in np.unique(cat_vid)))
    final = make_models(args.level)[best].fit(X, y)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "name": best, "features": feature_names, "level": args.level,
                 "window": float(d["window"]), "step": float(d["step"]), "seconds": float(d["seconds"]),
                 "cv": results[best][0], "threshold": threshold}, args.save)
    print(f"\nBest model: {best}. Retrained on all data and saved to {Path(args.save).resolve()}")

    if args.level == "video":   # the 8 features that separate real/fake best on their own
        sep = [abs(roc_auc_score(y, X[:, k]) - 0.5) if np.ptp(X[:, k]) > 0 else 0 for k in range(X.shape[1])]
        shown = [feature_names[k] for k in np.argsort(sep)[::-1][:8]]
    else:
        shown = ["coh_mean", "gcoh_mean", "corr_mean", "lag_mean", "bpm_std", "snr_mean", "snr_face", "hf_ratio"]
    plt = style()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    for ax, f in zip(axes.flat, shown):
        k = feature_names.index(f)
        lo, hi = np.percentile(X[:, k], [1, 99])
        bins = np.linspace(lo, hi, 30)
        ax.hist(X[y == 0, k], bins, alpha=0.6, density=True, label="real")
        ax.hist(X[y == 1, k], bins, alpha=0.6, density=True, label="fake")
        ax.set_title(f)
        ax.set_yticks([])
    axes[0, 0].legend()
    fig.suptitle(f"How real and fake {args.level}s differ, feature by feature", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig("outputs/detector_features.png")
    print(f"Saved {Path('outputs/detector_features.png').resolve()}")


if __name__ == "__main__":
    main()
