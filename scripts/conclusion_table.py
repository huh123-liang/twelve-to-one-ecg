"""The answer to "how does the model solve the problem": the pre-specified success criterion
applied to every diagnosis x lead set.

For every approach (= model variant + training strategy) and every reduced lead set:
  * predictions of all seeds are averaged (seed ensemble) to remove seed noise
  * reference = the 12-lead model of the same architecture family trained from scratch (S1)
  * per class: delta AUROC (new - reference) with a paired-bootstrap 95% CI, sensitivity at 90% specificity
  * verdict: screen        if the AUROC drop is non-inferior (CI lower bound of delta >= -0.02) AND sens@spec90 >= 0.80
             conditional   if only one of the two holds
             not_replaceable otherwise
  * sensitivity analysis: the same verdict with margins 0.01 and 0.03
T1 also gets a derived triage row: score = 1 - P(NORM), scored against the T2 label
(abnormal = any of MI/STTC/CD/HYP), so it is directly comparable with the dedicated T2 models.
The PRIMARY approach is chosen on the VALIDATION set (mean macro AUROC over the reduced lead sets),
never on the test set.

python scripts/conclusion_table.py --results results --out results/summary
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

from src.data import CLASSES, REDUCED_SETS  # noqa: E402
from src.metrics import NI_MARGIN, fast_auc, judge, macro_auroc, paired_bootstrap_per_class, sens_at_spec, sigmoid  # noqa: E402

FAMILY_REF = {"resnet1d": "resnet1d", "laresnet": "resnet1d", "bilstm": "bilstm", "vit": "vit"}

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/summary")
ap.add_argument("--n_boot", type=int, default=1000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# ---- load full-data runs, grouped by (task, approach, leads) ----
groups = defaultdict(list)
for mp in glob.glob(os.path.join(a.results, "*", "metrics.json")):
    m = json.load(open(mp))
    if m.get("train_fraction", 1.0) != 1.0 or m.get("strategy") == "B_baseline":
        continue
    z = np.load(os.path.join(os.path.dirname(mp), "preds.npz"))
    groups[(m.get("task", "t1"), m.get("variant", m["model"]), m.get("strategy", "S1_scratch"), m["model"], m["leads"])].append(z)


def ensemble(zs, key):
    return np.mean([sigmoid(z[f"{key}_logits"]) for z in zs], axis=0), zs[0][f"{key}_y"]


approaches = sorted({k[:4] for k in groups})
rows = []
val_scores = {}
for task, variant, strategy, model in approaches:
    ref_key = (task, FAMILY_REF[model], "S1_scratch", FAMILY_REF[model], "12lead")
    if ref_key not in groups:
        print(f"skip {variant}/{strategy}/{task}: no 12-lead reference {ref_key[1]}")
        continue
    p_ref, y = ensemble(groups[ref_key], "test")
    names = list(CLASSES) if task == "t1" else ["ABNORMAL"]
    if task == "t1":   # derived triage score; label = T2 definition (any abnormal superclass), NOT 1 - NORM
        # (NORM can co-occur with an abnormal superclass, so 1 - y_NORM != y_ABNORMAL for some records)
        y_abn = (groups[ref_key][0]["test_y5"][:, 1:].sum(1) > 0).astype(y.dtype)
        p_ref = np.concatenate([p_ref, 1 - p_ref[:, :1]], axis=1)
        y = np.concatenate([y, y_abn[:, None]], axis=1)
        names = names + ["ABNORMAL (derived 1-P(NORM))"]
    vals = []
    for leads in REDUCED_SETS:
        key = (task, variant, strategy, model, leads)
        if key not in groups:
            continue
        p_new, y_new = ensemble(groups[key], "test")
        pv, yv = ensemble(groups[key], "val")
        vals.append(macro_auroc(yv, pv))
        if task == "t1":
            p_new = np.concatenate([p_new, 1 - p_new[:, :1]], axis=1)
        deltas = paired_bootstrap_per_class(y, p_ref, p_new, n=a.n_boot)
        for k, c in enumerate(names):
            lo = deltas[k]["ci95"][0]
            s90 = sens_at_spec(y[:, k], p_new[:, k])
            rows.append({"task": task, "approach": f"{variant} [{strategy}]", "variant": variant, "strategy": strategy,
                         "leads": leads, "class": c, "n_seeds": len(groups[key]),
                         "auroc_12lead_ref": float(fast_auc(y[:, k], p_ref[:, k])),
                         "auroc_new": float(fast_auc(y[:, k], p_new[:, k])),
                         "delta_auroc": deltas[k]["diff"], "delta_ci_low": lo, "delta_ci_high": deltas[k]["ci95"][1],
                         "sens_at_spec90": s90,
                         "verdict": judge(lo, s90),
                         "verdict_margin_0.01": judge(lo, s90, margin=0.01),
                         "verdict_margin_0.03": judge(lo, s90, margin=0.03)})
    if vals:
        val_scores[(task, variant, strategy)] = (len(vals), float(np.mean(vals)))

if not rows:
    sys.exit("no reduced-lead runs with a 12-lead reference found")
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "conclusion_long.csv"), index=False)

symbol = {"screen": "✔ screen", "conditional": "△ conditional", "not_replaceable": "✘ not replaceable"}
md = ["# Conclusion table", "",
      f"Criterion (pre-specified): screen = AUROC drop non-inferior (95% CI lower bound of Δ ≥ −{NI_MARGIN}) "
      "AND sensitivity ≥ 0.80 at 90% specificity; conditional = one of the two; otherwise not replaceable.",
      "Each cell: verdict · ΔAUROC [95% CI] · sens@spec90. Predictions averaged over seeds.", ""]
# an approach can only be primary if it covers the most reduced lead sets (normally all five);
# among those, the best mean validation macro AUROC wins
val_tbl = pd.DataFrame([{"task": t, "approach": f"{v} [{s}]", "n_lead_sets": n, "val_mean_macro_auroc_reduced": sc}
                        for (t, v, s), (n, sc) in val_scores.items()]).sort_values(
    ["task", "n_lead_sets", "val_mean_macro_auroc_reduced"], ascending=[True, False, False])
val_tbl.to_csv(os.path.join(a.out, "approach_selection_val.csv"), index=False)
for task in sorted(df.task.unique()):
    vt = val_tbl[val_tbl.task == task]
    primary = vt.iloc[0]["approach"]
    md += [f"## Task {task.upper()} — primary approach (chosen on validation): {primary}", ""]
    sub = df[(df.task == task) & (df.approach == primary)]
    leads = [l for l in REDUCED_SETS if l in set(sub.leads)]
    md.append("| class | " + " | ".join(leads) + " |")
    md.append("|---|" + "---|" * len(leads))
    for c in sub["class"].unique():
        cells = []
        for l in leads:
            r = sub[(sub["class"] == c) & (sub.leads == l)].iloc[0]
            cells.append(f"{symbol[r.verdict]} · {r.delta_auroc:+.3f} [{r.delta_ci_low:+.3f}, {r.delta_ci_high:+.3f}] · "
                         f"{r.sens_at_spec90:.2f}")
        md.append(f"| {c} | " + " | ".join(cells) + " |")
    md += ["", "Validation ranking of approaches:", "", vt.to_markdown(index=False), ""]
    # sensitivity analysis summary
    for col in ["verdict", "verdict_margin_0.01", "verdict_margin_0.03"]:
        counts = sub[col].value_counts().to_dict()
        md.append(f"- {col}: {counts}")
    md.append("")
open(os.path.join(a.out, "conclusion_table.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))
