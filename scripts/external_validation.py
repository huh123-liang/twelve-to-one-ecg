"""E7: frozen-label external triage validation on Chapman-Shaoxing/Ningbo.

No model is fitted here. All checkpoints, scalers and operating thresholds come
from the existing PTB-XL runs and their training/validation folds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.io import loadmat  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from sklearn.metrics import average_precision_score, roc_curve  # noqa: E402
import torch  # noqa: E402
import wfdb  # noqa: E402

from src.data import LEADS, LEAD_SETS, Standardizer, apply_lead_mask, lead_indices, load_split, task_labels  # noqa: E402
from src.metrics import bootstrap_ci, ece_binary, fast_auc, sens_at_spec, sigmoid  # noqa: E402
from src.models import build_model, load_weights  # noqa: E402
from src.train import predict  # noqa: E402
from src.utils import get_device  # noqa: E402


# The original CR-3 mapping is retained verbatim for the v1 sensitivity result.
NORMAL_CODES = frozenset({"426783006", "426177001", "427084000", "427393009"})
MAPPING_V1 = {
    "MI": frozenset({"164865005"}),
    "STTC": frozenset({"429622005", "164930006", "428750005", "164931005", "164934002",
                       "59931005", "111975006"}),
    "CD": frozenset({"270492004", "195042002", "54016002", "28189009", "27885002",
                     "233917008", "698252002", "164909002", "59118001", "164947007",
                     "74390002", "195060002"}),
    "HYP": frozenset({"164873001", "89792004", "446358003"}),
}
# Mapping v2 adds the diagnoses listed in the official PhysioNet/CinC 2021
# evaluation-2021 dx_mapping_scored.csv and dx_mapping_unscored.csv. In the
# scored file CRBBB=RBBB (713427006=59118001) and CLBBB=LBBB
# (733534002=164909002); both members of each pair belong to CD here.
V2_ADDITIONS = {
    "CD": frozenset({"713427006", "713426002", "445118002", "445211001", "733534002",
                     "251120003", "6374002"}),
    "STTC": frozenset({"55930002", "413444003", "426434006", "164861001", "425419005"}),
    "MI": frozenset({"57054005", "54329005", "164867002"}),
    "HYP": frozenset({"67741000119109", "446813000"}),
}
MAPPING_V2 = {name: MAPPING_V1[name] | V2_ADDITIONS[name] for name in MAPPING_V1}
# Left ventricular high voltage is not equivalent to LVH; include it only in
# the separately named sensitivity analysis (v2_lvhv).
MAPPING_V2_LVHV = {**MAPPING_V2, "HYP": MAPPING_V2["HYP"] | {"55827005"}}
MAPPINGS = {"v1": MAPPING_V1, "v2": MAPPING_V2, "v2_lvhv": MAPPING_V2_LVHV}
SEEDS = (0, 1, 2)
CLASS_ORDER = ("NORM", "MI", "STTC", "CD", "HYP")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="directory containing the .hea/.mat records (WFDBRecords)")
    ap.add_argument("--name", default="chapman")
    ap.add_argument("--mapping", choices=tuple(MAPPINGS), default="v2")
    ap.add_argument("--ptb_cache", default="data/cache")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max_records", type=int, default=0, help="first N sorted headers; pilot only")
    return ap.parse_args(argv)


def read_codes(header: Path) -> set[str]:
    for line in header.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.match(r"#\s*Dx\s*:", line.strip(), flags=re.IGNORECASE):
            return {part.strip() for part in line.split(":", 1)[1].split(",") if part.strip()}
    return set()


def map_labels(codes: set[str], mapping: str = "v2"):
    """Normal requires only sinus codes; any mapped pathology makes a record abnormal."""
    mapped = {name for name, allowed in MAPPINGS[mapping].items() if codes & allowed}
    if mapped:
        return 1, np.array([0] + [int(name in mapped) for name in CLASS_ORDER[1:]], dtype=np.float32), mapped
    if codes and codes <= NORMAL_CODES:
        return 0, np.array([1, 0, 0, 0, 0], dtype=np.float32), {"NORM"}
    return None, None, set()


def code_names(root: Path) -> dict[str, str]:
    csv_path = root / "ConditionNames_SNOMED-CT.csv"
    if not csv_path.is_file():
        return {}
    table = pd.read_csv(csv_path, dtype=str)
    table.columns = [col.strip() for col in table.columns]
    if "Snomed_CT" not in table or "Full Name" not in table:
        raise ValueError(f"unexpected SNOMED name columns in {csv_path}: {list(table.columns)}")
    names = {}
    for code, frame in table.groupby("Snomed_CT"):
        names[str(code).strip()] = "; ".join(sorted(set(frame["Full Name"].dropna().str.strip())))
    return names


def code_category(code: str, mapping: str = "v2") -> str:
    if code in NORMAL_CODES:
        return "NORM"
    for name, allowed in MAPPINGS[mapping].items():
        if code in allowed:
            return name
    return "excluded_other"


def record_inventory(data_dir: Path, max_records: int, mapping: str = "v2"):
    headers = sorted(data_dir.rglob("*.hea"))
    if not headers:
        raise FileNotFoundError(f"no .hea headers under {data_dir}")
    if max_records:
        headers = headers[:max_records]
    rows, counts = [], Counter()
    for path in headers:
        rel = path.relative_to(data_dir).with_suffix("").as_posix()
        try:
            codes = read_codes(path)
            counts.update(codes)  # one record contributes at most one count per code
            triage, y5, mapped = map_labels(codes, mapping)
            status = "pending_signal" if triage is not None else "excluded_unmapped_dx"
        except Exception as exc:
            codes, triage, y5, mapped = set(), None, None, set()
            status = f"excluded_header_error:{type(exc).__name__}"
        rows.append({"record_id": rel, "header": str(path), "dx_codes": ",".join(sorted(codes)),
                     "mapped_classes": ",".join(sorted(mapped)), "triage_label": triage,
                     "status": status, "y5": y5})
    return rows, counts


def unit_to_mv(unit: str) -> float:
    unit = str(unit).strip().lower().replace("μ", "u").replace("µ", "u")
    if unit in ("mv", "millivolt", "millivolts"):
        return 1.0
    if unit in ("uv", "microvolt", "microvolts"):
        return 1e-3
    if unit in ("v", "volt", "volts"):
        return 1e3
    raise ValueError(f"unknown ECG physical unit {unit!r}")


def read_signal_mv(record_base: str):
    """WFDB physical samples, with scipy fallback for a damaged WFDB .mat reader."""
    try:
        signal, info = wfdb.rdsamp(record_base)
        names, units, fs = info["sig_name"], info["units"], info["fs"]
    except Exception:
        header = wfdb.rdheader(record_base)
        raw = loadmat(record_base + ".mat")["val"]
        if raw.ndim != 2 or raw.shape[0] != header.n_sig:
            raise ValueError("fallback .mat val shape disagrees with WFDB header")
        gain = np.asarray(header.adc_gain, dtype=np.float64)
        base = np.asarray(header.baseline, dtype=np.float64)
        if np.any(~np.isfinite(gain)) or np.any(gain == 0):
            raise ValueError("invalid ADC gain")
        signal = ((raw.astype(np.float64) - base[:, None]) / gain[:, None]).T
        names, units, fs = header.sig_name, header.units, header.fs
    names = [str(name).upper() for name in names]
    if len(names) != 12 or set(names) != set(LEADS):
        raise ValueError(f"lead set differs from PTB-XL: {names}")
    if abs(float(fs) - 500.0) > 1e-6:
        raise ValueError(f"expected 500 Hz, got {fs}")
    order = [names.index(name) for name in LEADS]
    signal = np.asarray(signal, dtype=np.float64)[:, order]
    scales = np.asarray([unit_to_mv(units[i]) for i in order], dtype=np.float64)
    return signal * scales[None, :]


def preprocess_signal(record_base: str):
    signal = read_signal_mv(record_base)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("nonfinite_signal")
    if np.any(np.ptp(signal, axis=0) <= 1e-8):
        raise ValueError("flatline_lead")
    # The filter runs before taking the first 10 seconds, as pre-specified.
    reduced = resample_poly(signal, 1, 5, axis=0)
    if not np.isfinite(reduced).all():
        raise ValueError("nonfinite_signal")
    reduced = reduced[:1000]
    if len(reduced) < 1000:
        reduced = np.pad(reduced, ((0, 1000 - len(reduced)), (0, 0)))
    return reduced.astype(np.float32)


def external_arrays(args, output_dir: Path):
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    rows, counts = record_inventory(data_dir, args.max_records, args.mapping)
    names = code_names(data_dir.parent)
    code_rows = [{"snomed_code": code, "record_count": count,
                  "mapped_class": code_category(code, args.mapping), "name": names.get(code, "")}
                 for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    pd.DataFrame(code_rows).to_csv(output_dir / "snomed_code_counts.csv", index=False)
    suffix = f"_first{args.max_records}" if args.max_records else "_full"
    # v1 keeps its original cache name and bytes; v2 can never load v1 labels.
    mapping_suffix = "" if args.mapping == "v1" else f"_{args.mapping}"
    cache_path = data_dir.parent / f"external_{args.name}_100hz{mapping_suffix}{suffix}.npz"
    status_cache = data_dir.parent / f"external_{args.name}_status{mapping_suffix}{suffix}.csv"
    if args.mapping != "v2_lvhv" and cache_path.is_file() and status_cache.is_file():
        with np.load(cache_path, allow_pickle=False) as packed:
            ids, X, y2, y5 = (packed[key].copy() for key in ("record_ids", "signals", "triage_y", "class_y"))
        statuses = pd.read_csv(status_cache)
        selected = statuses.loc[statuses.status == "included", "record_id"].astype(str).to_numpy()
        if not np.array_equal(ids, selected):
            raise RuntimeError("external cache IDs disagree with cached record status")
        print(f"reused {cache_path} ({len(ids)} included records)", flush=True)
    else:
        signals, labels2, labels5, ids = [], [], [], []
        base_signals, base_ids, base_status = None, {}, {}
        if args.mapping == "v2_lvhv":
            base_path = data_dir.parent / f"external_{args.name}_100hz_v2{suffix}.npz"
            base_status_path = data_dir.parent / f"external_{args.name}_status_v2{suffix}.csv"
            if not base_path.is_file() or not base_status_path.is_file():
                raise FileNotFoundError("run --mapping v2 first to build its complete signal cache")
            with np.load(base_path, allow_pickle=False) as packed:
                base_record_ids = packed["record_ids"].copy()
                base_signals = packed["signals"].copy()
            base_ids = {record_id: i for i, record_id in enumerate(base_record_ids)}
            base_status = dict(zip(pd.read_csv(base_status_path).record_id,
                                   pd.read_csv(base_status_path).status))
            cache_source = f"{base_path} plus newly mapped LVHV signals from {data_dir}"
        else:
            cache_source = str(cache_path)
        for i, row in enumerate(rows, 1):
            if row["status"] != "pending_signal":
                continue
            record_base = str(data_dir / row["record_id"])
            if row["record_id"] not in base_ids and not Path(record_base + ".mat").is_file():
                raise FileNotFoundError(f"mapped record is missing its .mat signal: {record_base}")
            try:
                if row["record_id"] in base_ids:
                    signal = base_signals[base_ids[row["record_id"]]]
                elif str(base_status.get(row["record_id"], "")).startswith("excluded_signal_"):
                    row["status"] = base_status[row["record_id"]]
                    continue
                else:
                    signal = preprocess_signal(record_base)
                signals.append(signal)
                labels2.append(row["triage_label"])
                labels5.append(row["y5"])
                ids.append(row["record_id"])
                row["status"] = "included"
            except Exception as exc:
                reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                row["status"] = "excluded_signal_" + reason.replace(" ", "_")[:80]
            if i % 1000 == 0:
                print(f"read {i}/{len(rows)} headers; included={len(ids)}", flush=True)
        if not ids:
            raise RuntimeError("no records survived the fixed label and signal filters")
        X = np.stack(signals)
        y2 = np.asarray(labels2, dtype=np.float32).reshape(-1, 1)
        y5 = np.stack(labels5)
        ids = np.asarray(ids, dtype=str)
        statuses = pd.DataFrame([{k: v for k, v in row.items() if k not in ("header", "y5")}
                                 for row in rows])
        statuses.to_csv(status_cache, index=False)
        if args.mapping != "v2_lvhv":
            np.savez_compressed(cache_path, record_ids=ids, signals=X, triage_y=y2, class_y=y5)
            print(f"cached {len(ids)} included records at {cache_path}", flush=True)
        else:
            print(f"assembled {len(ids)} records from v2 cache and LVHV additions", flush=True)
    shutil.copyfile(status_cache, output_dir / "record_status.csv")
    return ids, X, y2, y5, statuses, str(cache_path) if args.mapping != "v2_lvhv" else cache_source


def model_groups():
    groups = [("resnet1d", "t2", leads) for leads in ("12lead", "leadI")]
    groups += [("laresnet", "t2", leads) for leads in LEAD_SETS]
    groups += [(model, "t1", leads) for model in ("resnet1d", "bilstm")
               for leads in ("12lead", "leadI")]
    return groups


def run_folder(results: str, model: str, task: str, leads: str, seed: int) -> Path:
    suffix = "_t2" if task == "t2" else ""
    return Path(results) / f"{model}_{leads}{suffix}_frac1_seed{seed}"


def load_model(folder: Path, device):
    with (folder / "config.json").open(encoding="utf-8") as f:
        cfg = json.load(f)
    n_in = 24 if cfg["model"] == "laresnet" else len(LEAD_SETS[cfg["leads"]])
    n_classes = 1 if cfg["task"] == "t2" else 5
    model = build_model(cfg["model"], n_in, n_classes, kernel=cfg.get("kernel", 7),
                        residual=not cfg.get("no_residual", False), cell=cfg.get("cell", "lstm"))
    ckpt = folder.parent / cfg.get("checkpoint_run", folder.name) / "best.pt"
    load_weights(model, str(ckpt), map_location="cpu")
    return model.to(device).eval(), cfg


def predict_external(model, X, std12, std1, leads, is_la, device, batch):
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            raw = X[start:start + batch]
            if is_la:
                prepared = apply_lead_mask(std12(raw), leads)
            elif leads == "leadI":
                prepared = std1(raw[:, :, :1])
            else:
                prepared = std12(raw[:, :, lead_indices(leads)])
            out.append(predict(model, prepared, device, batch=batch))
    return np.concatenate(out)


def ensemble_predictions(args, model_name, task, leads, X, std12, std1, device, splits):
    ext, val, test = [], [], []
    for seed in SEEDS:
        folder = run_folder(args.results, model_name, task, leads, seed)
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        model, cfg = load_model(folder, device)
        if cfg["model"] != model_name or cfg["task"] != task or cfg["leads"] != leads or cfg["train_fraction"] != 1.0:
            raise RuntimeError(f"unexpected checkpoint config in {folder}")
        with np.load(folder / "preds.npz") as packed:
            if not np.array_equal(packed["test_ids"], splits["test"].ids):
                raise RuntimeError(f"internal test IDs changed in {folder}")
            val_logits = packed["val_logits"].copy()
            test_logits = packed["test_logits"].copy()
        external_logits = predict_external(model, X, std12, std1, leads, model_name == "laresnet", device, args.batch)
        if task == "t1":
            ext.append(1 - sigmoid(external_logits[:, :1]))
            val.append(1 - sigmoid(val_logits[:, :1]))
            test.append(1 - sigmoid(test_logits[:, :1]))
        else:
            ext.append(sigmoid(external_logits))
            val.append(sigmoid(val_logits))
            test.append(sigmoid(test_logits))
        print(f"{folder.name}: external predictions {len(X)}", flush=True)
        if task == "t1":
            # Keep the full five-class probabilities for the secondary analysis.
            yield_full = sigmoid(external_logits)
            if seed == SEEDS[0]:
                class_probabilities = []
            class_probabilities.append(yield_full)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = (np.mean(ext, axis=0), np.mean(val, axis=0), np.mean(test, axis=0))
    return out, (np.mean(class_probabilities, axis=0) if task == "t1" else None)


def scalar_metrics(y, p):
    yy, pp = np.asarray(y).ravel(), np.asarray(p).ravel()
    if len(np.unique(yy)) < 2:
        return {"auroc": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "auprc": np.nan, "sens_at_spec90": np.nan, "ece": np.nan}
    ci = bootstrap_ci(yy[:, None], pp[:, None], lambda a, b: fast_auc(a[:, 0], b[:, 0]), n=1000)
    return {"auroc": float(fast_auc(yy, pp)), "ci_low": ci[0], "ci_high": ci[1],
            "auprc": float(average_precision_score(yy, pp)),
            "sens_at_spec90": float(sens_at_spec(yy, pp)), "ece": float(ece_binary(yy, pp))}


def threshold_at_spec90(y_val, p_val):
    yy, pp = np.asarray(y_val).ravel(), np.asarray(p_val).ravel()
    if len(np.unique(yy)) < 2:
        raise RuntimeError("PTB-XL validation set lacks one of the triage classes")
    fpr, tpr, thresholds = roc_curve(yy, pp)
    permitted = np.flatnonzero(fpr <= 0.1 + 1e-12)
    chosen = permitted[np.argmax(tpr[permitted])]
    return float(thresholds[chosen])


def operating_metrics(y, p, threshold):
    yy, pred = np.asarray(y).ravel(), np.asarray(p).ravel() >= threshold
    positives, negatives = yy == 1, yy == 0
    return (float(pred[positives].mean()) if positives.any() else np.nan,
            float((~pred[negatives]).mean()) if negatives.any() else np.nan)


def write_figure(table, path):
    ordered = table.sort_values(["source_task", "model", "leads"]).reset_index(drop=True)
    names = [f"{r.model} {r.leads} ({r.source_task.upper()})" for r in ordered.itertuples()]
    yy = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, max(5, len(ordered) * 0.44)))
    ax.scatter(ordered.internal_ptb_test_auroc, yy - 0.13, label="PTB-XL test", color="#31598a", s=28)
    ax.errorbar(ordered.external_auroc, yy + 0.13,
                xerr=[ordered.external_auroc - ordered.external_ci_low,
                      ordered.external_ci_high - ordered.external_auroc],
                fmt="o", color="#bc602a", capsize=2, markersize=4, label="Chapman external (95% CI)")
    ax.set(yticks=yy, yticklabels=names, xlabel="Triage AUROC", xlim=(0, 1),
           title="Frozen PTB-XL models on Chapman-Shaoxing/Ningbo")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def add_mapping_comparison(summary_dir: Path, name: str):
    """Append the three fixed mapping analyses without changing archived v1 files."""
    paths = {
        "v1": summary_dir / f"external_{name}_v1_mapping",
        "v2": summary_dir / f"external_{name}",
        "v2_lvhv": summary_dir / f"external_{name}_v2_lvhv",
    }
    if not all((path / "record_status.csv").is_file() and (path / "e7_external_t2.csv").is_file()
               for path in paths.values()):
        return
    rows = []
    for mapping, path in paths.items():
        status = pd.read_csv(path / "record_status.csv")
        selected = status[status.status == "included"]
        table = pd.read_csv(path / "e7_external_t2.csv")
        row = {"mapping": mapping, "included": len(selected),
               "positive": int(selected.triage_label.sum()),
               "negative": int((selected.triage_label == 0).sum())}
        for model, leads, label in (("resnet1d", "12lead", "ResNet1d 12lead"),
                                    ("resnet1d", "leadI", "ResNet1d leadI"),
                                    ("laresnet", "12lead", "LA-ResNet 12lead"),
                                    ("laresnet", "leadI", "LA-ResNet leadI")):
            match = table[(table.model == model) & (table.source_task == "t2") & (table.leads == leads)]
            if len(match) != 1:
                raise RuntimeError(f"expected one {label} T2 row for {mapping}")
            row[f"{label} AUROC"] = float(match.iloc[0].external_auroc)
            row[f"{label} sens@spec90"] = float(match.iloc[0].external_sens_at_spec90)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    section = ("\n## Mapping sensitivity comparison\n\n"
               + comparison.round(4).to_markdown(index=False)
               + "\n\nv2 is the main analysis; v1 is the archived initial mapping, and v2_lvhv "
                 "additionally treats left ventricular high voltage as HYP. PTB-XL internal "
                 "test results and operating-threshold rules are identical across these rows.\n")
    for mapping in ("v2", "v2_lvhv"):
        report_path = paths[mapping] / "e7_external.md"
        if report_path.is_file():
            body = report_path.read_text(encoding="utf-8").split("\n## Mapping sensitivity comparison\n", 1)[0]
            report_path.write_text(body.rstrip() + "\n" + section, encoding="utf-8")


def run(args):
    started = time.time()
    if args.batch < 1 or args.max_records < 0:
        raise ValueError("batch must be positive and max_records nonnegative")
    suffix = f"/pilot_{args.max_records}" if args.max_records else ""
    mapping_suffix = {"v1": "_v1_mapping", "v2": "", "v2_lvhv": "_v2_lvhv"}[args.mapping]
    output_dir = Path(args.out) / f"external_{args.name}{mapping_suffix}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    ids, X, y_ext, y5_ext, statuses, cache_path = external_arrays(args, output_dir)
    splits = load_split(args.ptb_cache, 100, train_fraction=1.0, seed=0)
    y_val = task_labels(splits["val"].y5, "t2")
    y_test = task_labels(splits["test"].y5, "t2")
    std12 = Standardizer(splits["train"].X)
    std1 = Standardizer(splits["train"].X[:, :, :1])
    device = get_device(args.device)
    rows, class_rows = [], []
    for model_name, task, leads in model_groups():
        (p_ext, p_val, p_test), class_probs = ensemble_predictions(
            args, model_name, task, leads, X, std12, std1, device, splits)
        ext = scalar_metrics(y_ext, p_ext)
        internal = scalar_metrics(y_test, p_test)
        threshold = threshold_at_spec90(y_val, p_val)
        transported_sens, transported_spec = operating_metrics(y_ext, p_ext, threshold)
        row = {"model": model_name, "source_task": task, "leads": leads, "n_seeds": 3,
               "external_n": len(ids), "external_positive": int(y_ext.sum()),
               "external_auroc": ext["auroc"], "external_ci_low": ext["ci_low"],
               "external_ci_high": ext["ci_high"], "external_auprc": ext["auprc"],
               "external_sens_at_spec90": ext["sens_at_spec90"],
               "external_ece": ext["ece"] if task == "t2" else np.nan,
               "ptb_val_spec90_threshold": threshold,
               "external_sens_at_ptb_threshold": transported_sens,
               "external_spec_at_ptb_threshold": transported_spec,
               "internal_ptb_test_auroc": internal["auroc"],
               "internal_ptb_test_sens_at_spec90": internal["sens_at_spec90"],
               "delta_auroc_external_minus_internal": ext["auroc"] - internal["auroc"],
               "delta_sens_at_spec90_external_minus_internal":
                   ext["sens_at_spec90"] - internal["sens_at_spec90"]}
        rows.append(row)
        print(f"{model_name} {leads} {task}: external AUROC {ext['auroc']:.4f}, "
              f"PTB-XL AUROC {internal['auroc']:.4f}", flush=True)
        if class_probs is not None:
            for k, name in enumerate(CLASS_ORDER[1:], 1):
                count = int(y5_ext[:, k].sum())
                if count >= 30 and count < len(y5_ext):
                    scores = scalar_metrics(y5_ext[:, k], class_probs[:, k])
                    class_rows.append({"model": model_name, "leads": leads, "class": name,
                                       "positive_n": count, "external_n": len(ids),
                                       "external_auroc": scores["auroc"],
                                       "ci_low": scores["ci_low"], "ci_high": scores["ci_high"]})
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "e7_external_t2.csv", index=False)
    pd.DataFrame(class_rows, columns=["model", "leads", "class", "positive_n", "external_n",
                                      "external_auroc", "ci_low", "ci_high"]).to_csv(
                                          output_dir / "e7_external_t1_per_class.csv", index=False)
    plot_path = Path(args.out) / (f"fig11_external_{args.name}{mapping_suffix}"
                                      f"{'_pilot' if args.max_records else ''}.png")
    write_figure(table, plot_path)
    counts = statuses.status.value_counts().to_dict()
    report = [f"# E7: external Chapman-Shaoxing/Ningbo validation ({args.name}; {args.mapping} mapping)", "",
              f"Records scanned: {len(statuses)}; included: {len(ids)}; external positives: {int(y_ext.sum())}; "
              f"external negatives: {int((1-y_ext).sum())}.", "",
              "The selected SNOMED mapping excludes arrhythmia-only and other unmapped records. "
              "This is a restricted T2 triage target, not all possible cardiac abnormalities.", "",
              f"Signal/cache source: `{cache_path}`. All 36 checkpoints were frozen PTB-XL runs (three seeds per row).", "",
              "## Record status", "", pd.DataFrame(sorted(counts.items()), columns=["status", "records"]).to_markdown(index=False), "",
              "## T2 triage including scores derived from T1 NORM", "",
              table.round(4).to_markdown(index=False), "",
              "The threshold in each row was selected on PTB-XL fold 9 to reach at least 90% specificity "
              "and was applied unchanged to the external set. External AUROC CI uses 1,000 record bootstraps. "
              "External sensitivity at 90% specificity is a separate descriptive ROC quantity.", "",
              "## T1 mapped classes with at least 30 external positives", "",
              pd.DataFrame(class_rows).round(4).to_markdown(index=False) if class_rows else "None met the count rule.", "",
              f"Wall time: {time.time()-started:.1f} s. Figure: `{plot_path}`.", ""]
    (output_dir / "e7_external.md").write_text("\n".join(report), encoding="utf-8")
    if not args.max_records:
        add_mapping_comparison(Path(args.out), args.name)
    print(f"E7 outputs written to {output_dir}; wall time {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    run(parse_args())
