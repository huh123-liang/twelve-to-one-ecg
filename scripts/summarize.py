"""Aggregate all finished runs into tables, statistical tests and report figures.

python scripts/summarize.py --results results --out results/summary

Tables
  all_runs.csv               one row per run (every metric, per-class metrics, noise robustness)
  table_mean_std.csv         mean ± std over seeds for every configuration
  paired_bootstrap_tests.csv architecture / lead-set / strategy comparisons (seed-ensembled, Holm-adjusted p)
Figures
  fig1  architectures at 12 leads (+ baselines)        fig6  reliability diagram
  fig2  reduced leads, S1 from scratch                 fig7  training curves
  fig2b training strategies across lead sets            fig8  ROC curves 12-lead vs lead I
  fig3  per-class AUROC heatmap                         fig9  ablations (Lecture exercises)
  fig4  noise robustness                                fig10 T2 triage across lead sets
  fig5  label efficiency
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src.data import CLASSES  # noqa: E402
from src.metrics import paired_bootstrap, sigmoid  # noqa: E402

LEAD_ORDER = ["12lead", "limb6", "chest6", "2lead", "leadII", "leadI"]
ARCH = {"resnet1d": "ResNet1d", "bilstm": "CNN-BiLSTM", "vit": "ECG-ViT"}
STRAT = {"S1_scratch": "S1 from scratch", "S2_transfer_finetune": "S2 transfer (fine-tune)",
         "S2_transfer_frozen": "S2 transfer (frozen)", "S3_lead_agnostic": "S3 LA-ResNet"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/summary")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# ---------------------------------------------------------------- collect
rows = []
for p in glob.glob(os.path.join(a.results, "*", "metrics.json")):
    r = json.load(open(p))
    row = {k: r.get(k) for k in ["run", "model", "variant", "strategy", "task", "leads", "seed", "train_fraction",
                                 "n_train", "n_params", "macro_auroc", "macro_auprc", "macro_f1",
                                 "macro_sens_at_spec90", "ece", "ece_temp_scaled", "brier", "temperature",
                                 "epochs_run", "train_time_s"]}
    row["variant"] = row["variant"] or r["model"]          # v1 runs had no variant/strategy/task fields
    row["strategy"] = row["strategy"] or "S1_scratch"
    row["task"] = row["task"] or "t1"
    row["auroc_ci_lo"], row["auroc_ci_hi"] = r.get("macro_auroc_ci95", [np.nan, np.nan])
    for c, pc in r["per_class"].items():
        for k in ["auroc", "auprc", "f1", "sens_at_spec90", "ppv", "npv", "sensitivity", "specificity"]:
            row[f"{k}_{c}"] = pc.get(k)
    for k, v in r.get("noise_robustness", {}).items():
        row[f"noise_{k}"] = v
    row["dir"] = os.path.dirname(p)
    rows.append(row)
if not rows:
    sys.exit("no finished runs found")
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "all_runs.csv"), index=False)

metrics = ["macro_auroc", "macro_auprc", "macro_f1", "macro_sens_at_spec90", "ece", "ece_temp_scaled"]
agg = df.groupby(["task", "variant", "strategy", "leads", "train_fraction"])[metrics].agg(["mean", "std", "count"])
agg.to_csv(os.path.join(a.out, "table_mean_std.csv"))
print(agg.round(4).to_string())

t1 = df[(df.task == "t1") & (df.train_fraction == 1.0)]
s1 = t1[(t1.strategy == "S1_scratch") & t1.variant.isin(ARCH)]
d12 = s1[s1.leads == "12lead"]


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(os.path.join(a.out, name))
    plt.close(fig)


# ---------------------------------------------------------------- fig1 architectures
if len(d12):
    base = t1[(t1.strategy == "B_baseline") & (t1.leads == "12lead")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
    baseline_lines = []
    baseline_labels = []
    baseline_colors = {"b0-prior": "#555555", "b1-logreg": "#CC6677", "b2-features": "#228833"}
    for ax, m in zip(axes, ["macro_auroc", "macro_auprc", "macro_sens_at_spec90"]):
        g = d12.groupby("variant")[m].agg(["mean", "std"]).reindex([k for k in ARCH if k in set(d12.variant)])
        ax.bar([ARCH[i] for i in g.index], g["mean"], yerr=g["std"].fillna(0), capsize=3, color="#4C72B0")
        for _, b in base.iterrows():
            line = ax.axhline(b[m], ls="--", lw=1, color=baseline_colors.get(b["variant"], "#888"))
            if ax is axes[0]:
                baseline_lines.append(line)
                baseline_labels.append(b["variant"])
        ax.set_title(m.replace("macro_", "").replace("_", " ").upper())
        ax.tick_params(axis="x", rotation=15)
    if baseline_labels:   # no baselines yet (e.g. mock data) -> no legend; ncol=0 would crash
        fig.legend(baseline_lines, baseline_labels, loc="upper center", ncol=len(baseline_labels),
                   frameon=False, fontsize=7)
    save(fig, "fig1_architecture_12lead.png", rect=(0, 0, 1, 0.9))

# ---------------------------------------------------------------- fig2 reduced leads (S1)
if s1.leads.nunique() > 1:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for m in ARCH:
        g = s1[s1.variant == m].groupby("leads")["macro_auroc"].agg(["mean", "std"])
        g = g.reindex([l for l in LEAD_ORDER if l in g.index])
        if len(g):
            ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=3, label=ARCH[m])
    ax.set_ylabel("Test macro AUROC"); ax.set_xlabel("Lead configuration")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig2_reduced_leads.png")

# ---------------------------------------------------------------- fig2b strategies
strat = t1[t1.strategy.isin(STRAT) & t1.variant.isin(["resnet1d", "resnet1d-tfft", "resnet1d-tffrz", "laresnet"])]
if strat.strategy.nunique() > 1:
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    # numeric x positions in LEAD_ORDER (a categorical axis would order by first appearance), small
    # horizontal offsets so overlapping strategies (e.g. frozen vs LA-ResNet at lead I) stay visible
    markers = {"S1_scratch": "o", "S2_transfer_finetune": "s", "S2_transfer_frozen": "D", "S3_lead_agnostic": "^"}
    for j, (s, lab) in enumerate(STRAT.items()):
        g = strat[strat.strategy == s].groupby("leads")["macro_auroc"].agg(["mean", "std"])
        g = g.reindex([l for l in LEAD_ORDER if l in g.index])
        if len(g):
            xs = np.array([LEAD_ORDER.index(l) for l in g.index]) + (j - 1.5) * 0.06
            ax.errorbar(xs, g["mean"], yerr=g["std"].fillna(0), marker=markers.get(s, "o"), capsize=3, label=lab,
                        ms=6 if s == "S2_transfer_frozen" else 5)
    ax.set_xticks(range(len(LEAD_ORDER)), LEAD_ORDER)
    ax.set_ylabel("Test macro AUROC"); ax.set_xlabel("Lead configuration")
    ax.set_title("Strategies for reduced-lead models (ResNet backbone)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig2b_strategies.png")

# ---------------------------------------------------------------- fig3 per-class heatmap
hm = t1[t1.strategy.isin(["S1_scratch", "S3_lead_agnostic"]) & t1.variant.isin(list(ARCH) + ["laresnet"])]
g = hm.groupby(["variant", "leads"])[[f"auroc_{c}" for c in CLASSES]].mean()
if len(g):
    order = list(ARCH) + ["laresnet"]
    g = g.reindex(sorted(g.index, key=lambda t: (order.index(t[0]), LEAD_ORDER.index(t[1]) if t[1] in LEAD_ORDER else 99)))
    fig, ax = plt.subplots(figsize=(5.2, 0.3 * len(g) + 1.2))
    im = ax.imshow(g.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(g)), [f"{ARCH.get(m, 'LA-ResNet')} / {l}" for m, l in g.index])
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax.text(j, i, f"{g.values[i, j]:.3f}", ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, ax=ax, label="AUROC")
    save(fig, "fig3_per_class_auroc.png")

# ---------------------------------------------------------------- fig4 noise
ncols = [c for c in df.columns if c.startswith("noise_")]
if ncols and len(d12):
    snrs = sorted(ncols, key=lambda c: -float(c[6:-2]))
    fig, ax = plt.subplots(figsize=(4.8, 3))
    for m in ARCH:
        for l, ls in [("12lead", "-"), ("leadI", "--")]:
            sub = s1[(s1.variant == m) & (s1.leads == l)]
            if len(sub):
                ax.plot(["clean"] + [c[6:] for c in snrs], [sub["macro_auroc"].mean()] + [sub[c].mean() for c in snrs],
                        marker="o", ls=ls, label=f"{ARCH[m]} ({l})")
    ax.set_ylabel("Test macro AUROC"); ax.set_xlabel("Added white noise (SNR)")
    ax.legend(frameon=False, fontsize=6)
    save(fig, "fig4_noise_robustness.png")

# ---------------------------------------------------------------- fig5 label efficiency
le = df[(df.task == "t1") & df.leads.isin(["12lead", "leadI"])]
if le.train_fraction.nunique() > 1:
    fig, ax = plt.subplots(figsize=(7.5, 3.1))
    vit_colors = {}
    for (v, s, l), sub in le.groupby(["variant", "strategy", "leads"]):
        g = sub.groupby("train_fraction")["macro_auroc"].mean()
        if len(g) > 1:
            line, = ax.plot(g.index * 100, g.values, marker="o", ls="-" if l == "12lead" else "--",
                            label=f"{v} ({l})")
            if v == "vit":
                vit_colors[l] = line.get_color()
        elif v == "vit-msm" and len(g) == 1:
            ax.scatter(g.index * 100, g.values, marker="D", s=36, color=vit_colors[l],
                       label=f"vit-msm ({l}, 10%, 3 seeds)")
    ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
    ax.set_xlabel("Fraction of training patients"); ax.set_ylabel("Test macro AUROC")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "fig5_label_efficiency.png", rect=(0, 0, 0.7, 1))


def first_dir(sub):
    sub = sub.sort_values("seed")
    return sub.iloc[0]["dir"] if len(sub) else None


# ---------------------------------------------------------------- fig6 reliability, fig7 curves, fig8 ROC
fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.plot([0, 1], [0, 1], "k:", lw=1)
for m in ARCH:
    d = first_dir(d12[d12.variant == m])
    if d:
        z = np.load(os.path.join(d, "preds.npz"))
        p, y = sigmoid(z["test_logits"]).ravel(), z["test_y"].ravel()
        idx = np.clip(np.digitize(p, np.linspace(0, 1, 11)) - 1, 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(xs, ys, marker="o", label=ARCH[m])
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
ax.legend(frameon=False, fontsize=7)
save(fig, "fig6_reliability.png")

fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
for m in ARCH:
    d = first_dir(d12[d12.variant == m])
    if d and os.path.exists(os.path.join(d, "history.csv")):
        h = pd.read_csv(os.path.join(d, "history.csv"))
        axes[0].plot(h.epoch, h.train_loss, label=f"{ARCH[m]} train")
        axes[0].plot(h.epoch, h.val_loss, ls="--", label=f"{ARCH[m]} val")
        axes[1].plot(h.epoch, h.val_macro_auroc, label=ARCH[m])
axes[0].set_title("BCE loss"); axes[1].set_title("Validation macro AUROC")
for ax in axes:
    ax.set_xlabel("epoch")
axes[0].legend(frameon=False, fontsize=6); axes[1].legend(frameon=False, fontsize=7)
save(fig, "fig7_training_curves.png")

d_a, d_b = first_dir(d12[d12.variant == "resnet1d"]), first_dir(s1[(s1.variant == "resnet1d") & (s1.leads == "leadI")])
if d_a:
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(12, 2.6))
    for d, lab, ls in [(d_a, "12-lead", "-"), (d_b, "lead I", "--")]:
        if not d:
            continue
        z = np.load(os.path.join(d, "preds.npz"))
        p, y = sigmoid(z["test_logits"]), z["test_y"]
        for k, c in enumerate(CLASSES):
            if len(np.unique(y[:, k])) > 1:
                f, t, _ = roc_curve(y[:, k], p[:, k])
                axes[k].plot(f, t, ls=ls, label=lab)
    for k, c in enumerate(CLASSES):
        axes[k].plot([0, 1], [0, 1], "k:", lw=0.8); axes[k].axvline(0.1, color="#c33", lw=0.6)
        axes[k].set_title(c); axes[k].set_xlabel("1 - specificity")
    axes[0].set_ylabel("sensitivity"); axes[0].legend(frameon=False, fontsize=7)
    save(fig, "fig8_roc_12lead_vs_leadI.png")

# ---------------------------------------------------------------- fig9 ablations
AB_ORDER = ["resnet1d", "resnet1d-plain", "resnet1d-k3", "resnet1d-k15", "resnet1d-k31", "resnet1d-pw", "bilstm",
            "bilstm-rnn"]
ab = t1[(t1.leads == "12lead") & (t1.seed == 0) & (t1.train_fraction == 1.0) & t1.variant.isin(AB_ORDER)]
if ab.variant.nunique() > 2:
    g = ab.set_index("variant")[["macro_auroc", "macro_auprc"]]
    g = g.reindex([v for v in AB_ORDER if v in g.index])
    fig, ax = plt.subplots(figsize=(6.2, 3))
    xs = np.arange(len(g))
    ax.bar(xs - 0.2, g["macro_auroc"], 0.4, label="AUROC"); ax.bar(xs + 0.2, g["macro_auprc"], 0.4, label="AUPRC")
    ax.set_xticks(xs, g.index, rotation=25, fontsize=7); ax.set_ylim(max(0, g.values.min() - 0.1), 1)
    ax.set_title("Ablations, 12 leads, seed 0", fontsize=9); ax.legend(frameon=False, fontsize=7)
    save(fig, "fig9_ablations.png")

# ---------------------------------------------------------------- fig10 T2
t2 = df[(df.task == "t2") & (df.train_fraction == 1.0)]
if len(t2):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for (v, s), sub in t2.groupby(["variant", "strategy"]):
        g = sub.groupby("leads")[["macro_auroc", "macro_sens_at_spec90"]].mean()
        g = g.reindex([l for l in LEAD_ORDER if l in g.index])
        xs = [LEAD_ORDER.index(l) for l in g.index]   # fixed lead order, not order of first appearance
        axes[0].plot(xs, g["macro_auroc"], marker="o", label=f"{v} [{s}]")
        axes[1].plot(xs, g["macro_sens_at_spec90"], marker="o", label=f"{v} [{s}]")
    for ax in axes:
        ax.set_xticks(range(len(LEAD_ORDER)), LEAD_ORDER)
    axes[0].set_title("T2 triage AUROC"); axes[1].set_title("T2 sensitivity at 90% specificity")
    axes[1].axhline(0.8, color="#c33", lw=0.8, ls="--")
    axes[0].legend(frameon=False, fontsize=6)
    save(fig, "fig10_t2_triage.png")

# ---------------------------------------------------------------- paired bootstrap tests (seed 0)
tests = []


def load(variant, leads, seed=None, strategy="S1_scratch"):
    """Test labels and probabilities; seed=None averages all seeds (same seed ensemble as the conclusion table)."""
    sub = t1[(t1.variant == variant) & (t1.leads == leads) & (t1.strategy == strategy)]
    if seed is not None:
        sub = sub[sub.seed == seed]
    if not len(sub):
        return None
    zs = [np.load(os.path.join(d, "preds.npz")) for d in sub.sort_values("seed")["dir"]]
    return zs[0]["test_y"], np.mean([sigmoid(z["test_logits"]) for z in zs], axis=0)


def holm(pvals):
    """Holm-Bonferroni step-down adjustment (controls the family-wise error rate over all comparisons)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


names = list(ARCH)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        A, B = load(names[i], "12lead"), load(names[j], "12lead")
        if A and B:
            tests.append({"comparison": f"{names[i]} vs {names[j]} (12lead)", **paired_bootstrap(A[0], A[1], B[1])})
for m in names:
    A = load(m, "12lead")
    for l in LEAD_ORDER[1:]:
        B = load(m, l)
        if A and B:
            tests.append({"comparison": f"{m}: 12lead vs {l}", **paired_bootstrap(A[0], A[1], B[1])})
for l in LEAD_ORDER[1:]:
    A = load("resnet1d", l)
    for v, s in [("resnet1d-tfft", "S2_transfer_finetune"), ("resnet1d-tffrz", "S2_transfer_frozen"),
                 ("laresnet", "S3_lead_agnostic")]:
        B = load(v, l, strategy=s)
        if A and B:
            tests.append({"comparison": f"{l}: {v} vs resnet1d scratch", **paired_bootstrap(A[0], B[1], A[1])})
if tests:
    for t in tests:
        t["ci95_lo"], t["ci95_hi"] = t.pop("ci95")
    for t, q in zip(tests, holm([t["p_value"] for t in tests])):
        t["p_holm"] = float(q)
    pd.DataFrame(tests).to_csv(os.path.join(a.out, "paired_bootstrap_tests.csv"), index=False)
    print(pd.DataFrame(tests).round(4).to_string())
print("summary written to", a.out)
