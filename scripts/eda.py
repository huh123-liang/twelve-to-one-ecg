"""Dataset description figures/tables for the report (section 'Data').

python scripts/eda.py --ptbxl data/ptb-xl --cache data/cache --out results/eda
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import CLASSES, LEADS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="data/cache")
ap.add_argument("--sr", type=int, default=100)
ap.add_argument("--out", default="results/eda")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

meta = pd.read_csv(os.path.join(a.cache, f"meta_{a.sr}hz.csv"), index_col=0)
X = np.load(os.path.join(a.cache, f"signals_{a.sr}hz.npy"), mmap_mode="r")
meta["split"] = np.where(meta.strat_fold <= 8, "train", np.where(meta.strat_fold == 9, "val", "test"))

# Table 1: prevalence per split
t1 = meta.groupby("split")[CLASSES].agg(["sum", "mean"]).T.unstack()
t1.to_csv(os.path.join(a.out, "table1_label_prevalence.csv"))
summary = {
    "records": len(meta), "patients": meta.patient_id.nunique(),
    "age_median": float(meta.age[meta.age < 120].median()), "male_fraction": float((meta.sex == 0).mean()),
    "multi_label_fraction": float((meta[CLASSES].sum(axis=1) > 1).mean()),
}
pd.Series(summary).to_csv(os.path.join(a.out, "dataset_summary.csv"))
print(summary)

# Fig: label counts + co-occurrence
Y = meta[CLASSES].values
co = Y.T @ Y
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].bar(CLASSES, Y.sum(0), color="#4C72B0")
axes[0].set_title("Records per superclass")
im = axes[1].imshow(co, cmap="Blues")
axes[1].set_xticks(range(5), CLASSES)
axes[1].set_yticks(range(5), CLASSES)
for i in range(5):
    for j in range(5):
        axes[1].text(j, i, co[i, j], ha="center", va="center", fontsize=7)
axes[1].set_title("Label co-occurrence")
fig.tight_layout()
fig.savefig(os.path.join(a.out, "fig_labels.png"), dpi=150)
plt.close(fig)

# Fig: one example 12-lead ECG per class (single-label records)
fig, axes = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
t = np.arange(X.shape[1]) / a.sr
for k, c in enumerate(CLASSES):
    only = np.where((Y[:, k] == 1) & (Y.sum(axis=1) == 1))[0]
    if len(only):
        x = X[only[0]]
        for li, lead in enumerate(["I", "II", "V2", "V5"]):
            axes[k].plot(t, x[:, LEADS.index(lead)] - 2 * li, lw=0.6)
        axes[k].set_title(f"{c} (ecg_id {meta.index[only[0]]}); leads I, II, V2, V5", fontsize=8)
        axes[k].set_yticks([])
axes[-1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(os.path.join(a.out, "fig_examples.png"), dpi=150)
print("EDA written to", a.out)
