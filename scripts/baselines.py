"""E0: baselines every deep model must clearly beat (Lecture 5: "the baseline you must compute first").

B0 prior      : predict each class's training prevalence for every record (AUROC = 0.5)
B1 logistic   : StandardScaler + LogisticRegression on the down-sampled raw signal (Lecture 1 recipe),
                one binary classifier per class
B2 features   : StandardScaler + LogisticRegression on hand-crafted, SHIFT-INVARIANT per-lead features
                (CR-2). B1 is near chance because a fixed time sample means different things in different
                records (beats are not aligned); B2 summarises each lead with amplitude statistics, slope
                statistics, relative spectral band power and an autocorrelation rhythm estimate, which do
                not depend on where the beats fall. Written from scratch for BS6204 (no BS6221 code, K2).

Results are written in the same format as deep-learning runs (metrics.json + preds.npz), so
summarize.py and the conclusion table include them automatically.

python scripts/baselines.py                      # 12lead and leadI, tasks t1 and t2, B0 + B1 + B2
python scripts/baselines.py --models b2          # only B2 (does not touch existing B0/B1 results)
python scripts/baselines.py --cache data/mock-cache --out results_mock
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data import TASK_CLASSES, lead_indices, load_split, task_labels  # noqa: E402
from src.metrics import full_report, logit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="data/cache")
ap.add_argument("--sr", type=int, default=100)
ap.add_argument("--out", default="results")
ap.add_argument("--leads", nargs="+", default=["12lead", "leadI"])
ap.add_argument("--tasks", nargs="+", default=["t1", "t2"])
ap.add_argument("--pool", type=int, default=4, help="average-pool factor before the logistic regression")
ap.add_argument("--models", nargs="+", default=["b0", "b1", "b2"], choices=["b0", "b1", "b2"])
a = ap.parse_args()

splits = load_split(a.cache, a.sr)


def flat(X, li):
    X = X[:, :, li]
    N, T, C = X.shape
    T2 = T // a.pool * a.pool
    return X[:, :T2].reshape(N, T2 // a.pool, a.pool, C).mean(2).reshape(N, -1)


BANDS_HZ = [(0.5, 3), (3, 8), (8, 15), (15, 25), (25, 50)]
FEATURES_PER_LEAD = 13 + 2 + len(BANDS_HZ) + 2


def features(X, li, sr, chunk=2000):
    """Shift-invariant summary of every lead: (N, T, C) -> (N, C * FEATURES_PER_LEAD), in chunks to bound RAM."""
    return np.concatenate([_features(X[i:i + chunk], li, sr) for i in range(0, len(X), chunk)], axis=0)


def _features(X, li, sr):
    X = X[:, :, li].astype(np.float64)
    N, T, C = X.shape
    mu, sd = X.mean(1), X.std(1) + 1e-8
    z = (X - mu[:, None]) / sd[:, None]
    amp = [mu, sd, X.min(1), X.max(1)] + list(np.percentile(X, [1, 5, 25, 50, 75, 95, 99], axis=1)) + \
          [(z ** 3).mean(1), (z ** 4).mean(1)]                                  # 13 amplitude statistics
    d = np.diff(X, axis=1)
    slope = [np.abs(d).mean(1), d.std(1)]                                      # 2 slope statistics
    spec = np.abs(np.fft.rfft(X - mu[:, None], axis=1)) ** 2                   # (N, F, C) power spectrum
    freq = np.fft.rfftfreq(T, 1.0 / sr)
    total = spec[:, freq >= 0.5].sum(1) + 1e-12
    band = [spec[:, (freq >= lo) & (freq < hi)].sum(1) / total for lo, hi in BANDS_HZ]   # relative band power
    # autocorrelation (Wiener-Khinchin) -> dominant period between 0.3 s and 2.0 s ~ RR interval
    ac = np.fft.irfft(np.abs(np.fft.rfft(z, n=2 * T, axis=1)) ** 2, axis=1)[:, :T]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(0.3 * sr), int(2.0 * sr)
    lag = ac[:, lo:hi].argmax(1) + lo
    rhythm = [lag / sr, ac[:, lo:hi].max(1)]                                   # period (s), regularity
    F = np.stack(amp + slope + band + rhythm, axis=2)                          # (N, C, FEATURES_PER_LEAD)
    return F.reshape(N, -1)


def fit_logreg(Xtr, Xva, Xte, ytr, K):
    sc = StandardScaler().fit(Xtr)                     # fitted on TRAIN only (Lecture 1)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    val_l, test_l, n_par = np.zeros((len(Xva), K)), np.zeros((len(Xte), K)), 0
    for k in range(K):
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit(Xtr, ytr[:, k])
        val_l[:, k], test_l[:, k] = clf.decision_function(Xva), clf.decision_function(Xte)
        n_par += clf.coef_.size + 1
    return val_l, test_l, n_par


def save(name, leads, task, val_logits, test_logits, y, n_params, t):
    tag = f"{name}_{leads}{'' if task == 't1' else '_' + task}_frac1_seed0"
    d = os.path.join(a.out, tag)
    os.makedirs(d, exist_ok=True)
    rep = full_report(y["test"], test_logits, y["val"], val_logits, TASK_CLASSES[task])
    rep.update({"run": tag, "model": name, "variant": name, "strategy": "B_baseline", "task": task, "leads": leads,
                "seed": 0, "train_fraction": 1.0, "n_train": int(len(y["train"])), "n_params": n_params,
                "epochs_run": 0, "train_time_s": t, "noise_robustness": {}})
    json.dump({"model": name, "leads": leads, "task": task, "seed": 0, "cache": a.cache, "sr": a.sr,
               "strategy": "B_baseline", "variant": name}, open(os.path.join(d, "config.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(d, "preds.npz"), val_logits=val_logits, val_y=y["val"], test_logits=test_logits,
                        test_y=y["test"], test_ids=splits["test"].ids, test_age=splits["test"].age,
                        test_sex=splits["test"].sex, test_y5=splits["test"].y5)
    json.dump(rep, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    print(f"[{tag}] macro AUROC {rep['macro_auroc']:.4f}  AUPRC {rep['macro_auprc']:.4f}  "
          f"sens@spec90 {rep['macro_sens_at_spec90']:.4f}")


for task in a.tasks:
    y = {k: task_labels(v.y5, task) for k, v in splits.items()}
    K = y["train"].shape[1]
    prior = y["train"].mean(0)
    for leads in a.leads:
        li = lead_indices(leads)
        if "b0" in a.models:
            # B0: the same constant (training prevalence) for every record -> AUROC exactly 0.5
            make = lambda n: logit(np.tile(prior, (n, 1)))
            save("b0-prior", leads, task, make(len(y["val"])), make(len(y["test"])), y, K, 0.0)
        if "b1" in a.models:
            # B1: logistic regression on the down-sampled signal (pre-specified, kept unchanged)
            t0 = time.time()
            val_l, test_l, n_par = fit_logreg(*(flat(splits[s].X, li) for s in ("train", "val", "test")),
                                              y["train"], K)
            save("b1-logreg", leads, task, val_l, test_l, y, n_par, time.time() - t0)
        if "b2" in a.models:
            # B2: logistic regression on shift-invariant per-lead features (CR-2)
            t0 = time.time()
            val_l, test_l, n_par = fit_logreg(*(features(splits[s].X, li, a.sr) for s in ("train", "val", "test")),
                                              y["train"], K)
            save("b2-features", leads, task, val_l, test_l, y, n_par, time.time() - t0)
