"""Evaluation metrics for multi-label ECG classification.

Discrimination : macro / per-class AUROC and AUPRC (Lecture 4: roc_auc_score)
Operating point: sensitivity at 90% specificity (part of the pre-specified success criterion)
Decision       : per-class thresholds chosen on VALIDATION to maximise F1, then applied to test:
                 F1, sensitivity, specificity, PPV, NPV and the confusion matrix (Lectures 3-4)
Calibration    : expected calibration error (ECE) and Brier score, before/after temperature scaling
Uncertainty    : record-level bootstrap 95% CI on the test set; paired bootstrap between two models
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

from .data import CLASSES

SPEC_TARGET = 0.90   # operating point used by the success criterion
SENS_TARGET = 0.80
NI_MARGIN = 0.02     # non-inferiority margin for delta AUROC


def sigmoid(z):
    # scipy's expit is the numerically stable logistic: same values, no overflow warning for large |z|
    return expit(z)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _per_class(y, p, fn):
    out = []
    for k in range(y.shape[1]):
        out.append(np.nan if len(np.unique(y[:, k])) < 2 else fn(y[:, k], p[:, k]))
    return np.array(out, dtype=float)


def fast_auc(y, s):
    """Mann-Whitney AUROC (ties averaged); ~20x faster than sklearn, used inside bootstraps."""
    from scipy.stats import rankdata
    pos = y == 1
    n1 = pos.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro_auroc(y, p):
    return float(np.nanmean([fast_auc(y[:, k], p[:, k]) for k in range(y.shape[1])]))


def sens_at_spec(y, s, spec=SPEC_TARGET):
    """Highest sensitivity reachable while specificity >= `spec` (read off the ROC curve)."""
    if len(np.unique(y)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def tune_thresholds(y_val, p_val, grid=np.linspace(0.05, 0.95, 91)):
    th = []
    for k in range(y_val.shape[1]):
        scores = [f1_score(y_val[:, k], p_val[:, k] >= t, zero_division=0) for t in grid]
        th.append(grid[int(np.argmax(scores))])
    return np.array(th)


def confusion_stats(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "sensitivity": div(tp, tp + fn),
            "specificity": div(tn, tn + fp), "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn)}


def ece_binary(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def macro_ece(y, p, n_bins=15):
    return float(np.mean([ece_binary(y[:, k], p[:, k], n_bins) for k in range(y.shape[1])]))


def fit_temperature(logits_val, y_val, iters=200):
    """One scalar temperature fitted on validation logits by minimising BCE (Guo et al., 2017)."""
    z = torch.tensor(logits_val, dtype=torch.float64)
    y = torch.tensor(y_val, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)
    lossf = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def full_report(y_test, logits_test, y_val, logits_val, class_names=None, bootstrap=True):
    """All test-set metrics. Thresholds and temperature come from the validation set only."""
    class_names = class_names or CLASSES
    p_val, p_test = sigmoid(logits_val), sigmoid(logits_test)
    th = tune_thresholds(y_val, p_val)
    pred = (p_test >= th).astype(int)
    T = fit_temperature(logits_val, y_val)
    p_test_ts = sigmoid(logits_test / T)

    auroc_k = _per_class(y_test, p_test, roc_auc_score)
    auprc_k = _per_class(y_test, p_test, average_precision_score)
    sens90_k = _per_class(y_test, p_test, sens_at_spec)
    f1_k = np.array([f1_score(y_test[:, k], pred[:, k], zero_division=0) for k in range(y_test.shape[1])])
    rep = {
        "macro_auroc": float(np.nanmean(auroc_k)),
        "macro_auprc": float(np.nanmean(auprc_k)),
        "macro_f1": float(f1_k.mean()),
        "macro_sens_at_spec90": float(np.nanmean(sens90_k)),
        "ece": macro_ece(y_test, p_test),
        "ece_temp_scaled": macro_ece(y_test, p_test_ts),
        "brier": float(((p_test - y_test) ** 2).mean()),
        "temperature": T,
        "thresholds": th.tolist(),
        "per_class": {},
    }
    for i, c in enumerate(class_names):
        rep["per_class"][c] = {"auroc": float(auroc_k[i]), "auprc": float(auprc_k[i]), "f1": float(f1_k[i]),
                               "sens_at_spec90": float(sens90_k[i]), **confusion_stats(y_test[:, i], pred[:, i])}
    if bootstrap:
        lo, hi = bootstrap_ci(y_test, p_test, macro_auroc)
        rep["macro_auroc_ci95"] = [lo, hi]
    return rep


def bootstrap_ci(y, p, fn, n=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    N = len(y)
    vals = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        vals.append(fn(y[i], p[i]))
    vals = np.array(vals)
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def paired_bootstrap(y, p_a, p_b, fn=macro_auroc, n=1000, seed=0):
    """Difference fn(A) - fn(B) with 95% CI and a two-sided bootstrap p-value."""
    rng = np.random.default_rng(seed)
    N = len(y)
    d = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        d.append(fn(y[i], p_a[i]) - fn(y[i], p_b[i]))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(fn(y, p_a) - fn(y, p_b)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_value": float(min(1.0, p))}


def paired_bootstrap_per_class(y, p_ref, p_new, n=1000, seed=0):
    """Per-class delta AUROC = AUROC(new) - AUROC(ref) with 95% CI, using the same resamples for all classes."""
    rng = np.random.default_rng(seed)
    N, K = y.shape
    d = np.full((n, K), np.nan)
    for b in range(n):
        i = rng.integers(0, N, N)
        for k in range(K):
            d[b, k] = fast_auc(y[i, k], p_new[i, k]) - fast_auc(y[i, k], p_ref[i, k])
    out = []
    for k in range(K):
        out.append({"diff": float(fast_auc(y[:, k], p_new[:, k]) - fast_auc(y[:, k], p_ref[:, k])),
                    "ci95": [float(np.nanpercentile(d[:, k], 2.5)), float(np.nanpercentile(d[:, k], 97.5))]})
    return out


def judge(delta_ci_low: float, sens90: float, margin: float = NI_MARGIN, sens_target: float = SENS_TARGET) -> str:
    """Pre-specified success criterion.

    Non-inferior if the AUROC drop (ref - new) has its upper 95% bound <= margin, i.e. the lower bound of
    (new - ref) >= -margin. 'screen' = non-inferior AND sensitivity at 90% specificity >= target.
    """
    ni = delta_ci_low >= -margin
    ok = sens90 >= sens_target
    if ni and ok:
        return "screen"
    if ni or ok:
        return "conditional"
    return "not_replaceable"
