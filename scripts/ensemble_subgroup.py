"""E6: ensembles and subgroup (fairness) analysis - no training needed, uses saved predictions.

1. Seed ensemble   : average the probabilities of the 3 seeds of the same configuration (Lecture 8)
2. Heterogeneous   : average ResNet1d + CNN-BiLSTM + ECG-ViT (Lecture 8, exercise 3) and look at the
                     records where the three architectures disagree most
3. Subgroups       : macro / per-class AUROC by sex and age band for the main models

python scripts/ensemble_subgroup.py --results results --out results/summary
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import CLASSES, LEAD_SETS  # noqa: E402
from src.metrics import fast_auc, macro_auroc, macro_ece, sigmoid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/summary")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

groups = defaultdict(list)
for mp in glob.glob(os.path.join(a.results, "*", "metrics.json")):
    m = json.load(open(mp))
    if m.get("train_fraction", 1.0) != 1.0 or m.get("strategy") == "B_baseline":
        continue
    z = dict(np.load(os.path.join(os.path.dirname(mp), "preds.npz")))
    groups[(m.get("task", "t1"), m.get("variant", m["model"]), m.get("strategy", "S1_scratch"), m["leads"])].append(z)

md = ["# Ensembles and subgroups", ""]

# ---- 1. seed ensemble ----
rows = []
for (task, variant, strategy, leads), zs in sorted(groups.items()):
    if len(zs) < 2:
        continue
    y = zs[0]["test_y"]
    ps = [sigmoid(z["test_logits"]) for z in zs]
    pe = np.mean(ps, axis=0)
    rows.append({"task": task, "variant": variant, "strategy": strategy, "leads": leads, "n_seeds": len(zs),
                 "single_auroc_mean": np.mean([macro_auroc(y, p) for p in ps]), "ensemble_auroc": macro_auroc(y, pe),
                 "single_ece_mean": np.mean([macro_ece(y, p) for p in ps]), "ensemble_ece": macro_ece(y, pe)})
if rows:
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "seed_ensemble.csv"), index=False)
    md += ["## Seed ensemble", "", t.round(4).to_markdown(index=False), ""]

# ---- 2. heterogeneous ensemble ----
rows, dis_rows = [], []
for leads in LEAD_SETS:
    keys = [("t1", v, "S1_scratch", leads) for v in ["resnet1d", "bilstm", "vit"]]
    if not all(k in groups for k in keys):
        continue
    y = groups[keys[0]][0]["test_y"]
    per_arch = [np.mean([sigmoid(z["test_logits"]) for z in groups[k]], axis=0) for k in keys]
    pe = np.mean(per_arch, axis=0)
    r = {"leads": leads, **{f"{k[1]}_auroc": macro_auroc(y, p) for k, p in zip(keys, per_arch)},
         "hetero_ensemble_auroc": macro_auroc(y, pe), "hetero_ensemble_ece": macro_ece(y, pe)}
    rows.append(r)
    dis = np.std(per_arch, axis=0).mean(1)                 # disagreement per record
    hi = dis >= np.quantile(dis, 0.8)
    err = (np.abs((pe >= 0.5).astype(float) - y)).mean(1)  # fraction of wrong labels per record
    dis_rows.append({"leads": leads, "error_rate_top20pct_disagreement": float(err[hi].mean()),
                     "error_rate_rest": float(err[~hi].mean()),
                     "auroc_top20pct_disagreement": macro_auroc(y[hi], pe[hi]),
                     "auroc_rest": macro_auroc(y[~hi], pe[~hi])})
if rows:
    t = pd.DataFrame(rows)
    d = pd.DataFrame(dis_rows)
    t.to_csv(os.path.join(a.out, "hetero_ensemble.csv"), index=False)
    d.to_csv(os.path.join(a.out, "disagreement.csv"), index=False)
    md += ["## Heterogeneous ensemble (ResNet1d + CNN-BiLSTM + ECG-ViT)", "", t.round(4).to_markdown(index=False), "",
           "Records where the architectures disagree most (top 20%) vs the rest:", "",
           d.round(4).to_markdown(index=False), ""]

# ---- 3. subgroups ----
def bands(age):
    return np.where(age < 50, "<50", np.where(age <= 70, "50-70", ">70"))


rows = []
targets = [("t1", "resnet1d", "S1_scratch"), ("t1", "laresnet", "S3_lead_agnostic"), ("t1", "vit", "S1_scratch"),
           ("t1", "bilstm", "S1_scratch")]
for task, variant, strategy in targets:
    for leads in ["12lead", "leadI"]:
        key = (task, variant, strategy, leads)
        if key not in groups:
            continue
        zs = groups[key]
        y = zs[0]["test_y"]
        p = np.mean([sigmoid(z["test_logits"]) for z in zs], axis=0)
        sex = zs[0]["test_sex"]
        age = zs[0]["test_age"].astype(float)
        sub = {"sex=male": sex == 0, "sex=female": sex == 1}
        for b in ["<50", "50-70", ">70"]:
            sub[f"age {b}"] = bands(age) == b
        for name, msk in sub.items():
            if msk.sum() < 10:
                continue
            r = {"approach": f"{variant} [{strategy}]", "leads": leads, "subgroup": name, "n": int(msk.sum()),
                 "macro_auroc": macro_auroc(y[msk], p[msk])}
            for k, c in enumerate(CLASSES):
                r[f"auroc_{c}"] = fast_auc(y[msk, k], p[msk, k])
            rows.append(r)
if rows:
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.out, "subgroups.csv"), index=False)
    md += ["## Subgroups (seed-ensembled predictions)", "", t.round(4).to_markdown(index=False), ""]

open(os.path.join(a.out, "ensemble_subgroup.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md[:60]))
