"""E8: two unlabeled ViT pretrains, then 12 matched 10%-label fine-tunes.

The same seed selects the same patient subset for scratch and MSM-initialized runs.
Completed pretrains (encoder.pt) and supervised runs (metrics.json) are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.metrics import bootstrap_ci, macro_auroc, paired_bootstrap, sigmoid  # noqa: E402

LEADS = ("12lead", "leadI")
SEEDS = (0, 1, 2)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reference_results", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pretrain_epochs", type=int, default=100)
    ap.add_argument("--pretrain_batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    return ap.parse_args(argv)


def pretrain_dir(out, leads):
    return os.path.join(out, f"vit-msm-pretrain_{leads}_seed0")


def run_dir(out, leads, seed, msm):
    variant = "vit-msm" if msm else "vit"
    return os.path.join(out, f"{variant}_{leads}_frac0.1_seed{seed}")


def job_list(args):
    jobs = []
    for leads in LEADS:
        done = os.path.exists(os.path.join(pretrain_dir(args.out, leads), "encoder.pt"))
        jobs.append(("pretrain", leads, 0, False, done))
    for leads in LEADS:
        for seed in SEEDS:
            for msm in (False, True):
                done = os.path.exists(os.path.join(run_dir(args.out, leads, seed, msm), "metrics.json"))
                jobs.append(("finetune", leads, seed, msm, done))
    return jobs


def command_for(args, kind, leads, seed, msm):
    warning_args = ["-W", "error::RuntimeWarning"] if any("error::RuntimeWarning" in x for x in sys.warnoptions) else []
    if kind == "pretrain":
        return [sys.executable, *warning_args, "scripts/pretrain_msm.py", "--leads", leads, "--seed", "0",
                "--cache", args.cache, "--out", args.out, "--epochs", str(args.pretrain_epochs),
                "--batch", str(args.pretrain_batch), "--device", args.device]
    cmd = [sys.executable, *warning_args, "-m", "src.train", "--model", "vit", "--leads", leads,
           "--seed", str(seed), "--train_fraction", "0.1", "--cache", args.cache,
           "--out", args.out, "--epochs", str(args.epochs), "--batch", str(args.batch),
           "--device", args.device, "--no_tensorboard"]
    if msm:
        cmd += ["--msm_from", pretrain_dir(args.out, leads)]
    return cmd


def complete(args, kind, leads, seed, msm):
    if kind == "pretrain":
        path = pretrain_dir(args.out, leads)
        return all(os.path.isfile(os.path.join(path, x)) for x in
                   ("config.json", "history.csv", "encoder.pt", "pretrain_summary.json", "reconstruction.png")) \
            and not os.path.exists(os.path.join(path, "metrics.json"))
    path = run_dir(args.out, leads, seed, msm)
    required = ("config.json", "history.csv", "best.pt", "preds.npz", "metrics.json")
    if not all(os.path.isfile(os.path.join(path, x)) for x in required):
        return False
    with open(os.path.join(path, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    with np.load(os.path.join(path, "preds.npz")) as z:
        ok = np.isfinite(z["test_logits"]).all() and np.isfinite(z["val_logits"]).all()
    return bool(ok and np.isfinite(metrics["macro_auroc"]))


def load_predictions(path):
    with np.load(os.path.join(path, "preds.npz")) as z:
        return z["test_y"], z["test_ids"], sigmoid(z["test_logits"])


def aligned_predictions(paths):
    records = [load_predictions(path) for path in paths]
    y, ids = records[0][:2]
    if not all(np.array_equal(y, item[0]) and np.array_equal(ids, item[1]) for item in records[1:]):
        raise RuntimeError("test labels or record ordering differ across E8 seeds/methods")
    return y, [item[2] for item in records]


def report(args):
    summary_dir = os.path.join(args.out, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    rows, paired = [], []
    for leads in LEADS:
        paths = {method: [run_dir(args.out, leads, seed, method == "msm") for seed in SEEDS]
                 for method in ("scratch", "msm")}
        if not all(complete(args, "finetune", leads, seed, method == "msm")
                   for method in paths for seed in SEEDS):
            raise RuntimeError(f"E8 report needs all six 10%-label runs for {leads}")
        y, all_probs = aligned_predictions(paths["scratch"] + paths["msm"])
        probs = {"scratch": all_probs[:3], "msm": all_probs[3:]}
        ensemble = {key: np.mean(ps, axis=0) for key, ps in probs.items()}
        reference_paths = [os.path.join(args.reference_results, f"vit_{leads}_frac1_seed{s}") for s in SEEDS]
        if all(os.path.exists(os.path.join(path, "preds.npz")) for path in reference_paths):
            y_ref, ref_probs = aligned_predictions(reference_paths)
            if not np.array_equal(y, y_ref):
                raise RuntimeError(f"E8 versus full-data ViT test labels differ for {leads}")
            reference_auc = macro_auroc(y, np.mean(ref_probs, axis=0))
        else:
            reference_auc = float("nan")  # mock smoke tests may lack full-data references
        for method in ("scratch", "msm"):
            scores = [macro_auroc(y, p) for p in probs[method]]
            auc = macro_auroc(y, ensemble[method])
            lo, hi = bootstrap_ci(y, ensemble[method], macro_auroc, n=1000)
            rows.append({"leads": leads, "method": method, "train_fraction": 0.1, "n_seeds": 3,
                         "seed_mean_auroc": float(np.mean(scores)), "seed_std_auroc": float(np.std(scores, ddof=1)),
                         "ensemble_auroc": auc, "ensemble_ci_low": lo, "ensemble_ci_high": hi,
                         "full_data_vit_reference_auroc": reference_auc,
                         **{f"seed{s}_auroc": scores[s] for s in SEEDS}})
        test = paired_bootstrap(y, ensemble["msm"], ensemble["scratch"], n=1000)
        paired.append({"leads": leads, "comparison": "msm_minus_scratch", "diff": test["diff"],
                       "ci95_low": test["ci95"][0], "ci95_high": test["ci95"][1], "p_value": test["p_value"],
                       **{f"seed{s}_diff": macro_auroc(y, probs["msm"][s]) - macro_auroc(y, probs["scratch"][s])
                          for s in SEEDS}})
    table = pd.DataFrame(rows)
    tests = pd.DataFrame(paired)
    table.to_csv(os.path.join(summary_dir, "e8_self_supervised.csv"), index=False)
    tests.to_csv(os.path.join(summary_dir, "e8_paired_tests.csv"), index=False)
    pretrains = []
    for leads in LEADS:
        with open(os.path.join(pretrain_dir(args.out, leads), "pretrain_summary.json"), encoding="utf-8") as f:
            pretrains.append({"leads": leads, **json.load(f)})
    markdown = ["# E8: masked-signal pretraining with 10% labeled PTB-XL data", "",
                "The two ECG-ViT encoders were pretrained without labels on folds 1-8; fold 9 selected the reconstruction checkpoint. Both supervised methods use the same seed-specific 10% patient subset.", "",
                "## Reconstruction pretraining", "", pd.DataFrame(pretrains).round(5).to_markdown(index=False), "",
                "## Three-seed classification results", "", table.round(5).to_markdown(index=False), "",
                "## Paired test on seed-ensemble predictions", "", tests.round(5).to_markdown(index=False), "",
                "The difference is MSM minus scratch. The 95% CI and two-sided p value use 1,000 paired record bootstraps; a displayed zero p value means below bootstrap resolution, not an exact probability of zero.", "",
                "The full-data ViT reference comes from the already completed E1/E2 runs and was not retrained.", ""]
    with open(os.path.join(summary_dir, "e8_self_supervised.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(markdown))
    print(f"E8 report written to {summary_dir}")


def main(args):
    jobs = job_list(args)
    for i, (kind, leads, seed, msm, done) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {kind} {leads} seed{seed} {'msm' if msm else 'scratch'}: "
              f"{'skip' if done else 'todo'}", flush=True)
    if args.list:
        return
    if not args.report_only:
        for i, (kind, leads, seed, msm, done) in enumerate(jobs, 1):
            if done and complete(args, kind, leads, seed, msm):
                continue
            print(f"[{i}/{len(jobs)}] run {kind} {leads} seed{seed} {'msm' if msm else 'scratch'}", flush=True)
            result = subprocess.run(command_for(args, kind, leads, seed, msm), check=False)
            is_complete = complete(args, kind, leads, seed, msm)
            if result.returncode:
                print(f"[{i}/{len(jobs)}] exit_code={result.returncode}; outputs_complete={is_complete}", flush=True)
            if not is_complete:
                raise RuntimeError(f"E8 output incomplete: {kind} {leads} seed{seed} msm={msm}")
    report(args)


if __name__ == "__main__":
    main(parse_args())
