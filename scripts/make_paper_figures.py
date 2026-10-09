"""Build the shared report/PPT figures and tables from completed result files.

This script reads existing experiment outputs only. It writes PNG/PDF figures
and CSV/Markdown tables to results/summary/paper_figs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.metrics import bootstrap_ci, macro_auroc, sigmoid  # noqa: E402


LEADS = ["12lead", "limb6", "chest6", "2lead", "leadII", "leadI"]
REDUCED = LEADS[1:]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
COLORS = {
    "resnet1d": "#2878B5",  # blue
    "bilstm": "#238B45",    # green
    "vit": "#7952A2",       # purple
    "laresnet": "#C44545",  # red
    "b2-features": "#686868",  # grey
}
LABELS = {
    "resnet1d": "ResNet1d", "bilstm": "CNN-BiLSTM", "vit": "ECG-ViT",
    "laresnet": "LA-ResNet", "b2-features": "B2 features",
}
LEAD_LABELS = ["12 leads", "limb 6", "chest 6", "I + II", "lead II", "lead I"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
    "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 120, "savefig.dpi": 300,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required result file missing: {path}")
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_table(frame: pd.DataFrame, out: Path, stem: str, note: str) -> None:
    frame.to_csv(out / f"{stem}.csv", index=False)
    rendered = frame.round(4).fillna("—")
    text = f"# {stem.replace('_', ' ')}\n\n{note}\n\n" + rendered.to_markdown(index=False) + "\n"
    (out / f"{stem}.md").write_text(text, encoding="utf-8")


def full_t1(all_runs: pd.DataFrame) -> pd.DataFrame:
    return all_runs[(all_runs.task == "t1") & (all_runs.train_fraction == 1.0)]


def fig_a(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set(xlim=(0, 12), ylim=(0, 5.2))
    ax.axis("off")

    def box(x, y, width, height, label, color="#EDF3F8"):
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08",
                               facecolor=color, edgecolor="#364453", linewidth=1.1)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, color="#52606D"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, linewidth=1.4, color=color))

    boxes = [
        (0.15, 2.15, 1.25, 0.8, "PTB-XL\nECG records"),
        (1.72, 2.15, 1.55, 0.8, "Official folds\ntrain / val / test"),
        (3.58, 2.15, 1.35, 0.8, "Six lead\nconfigurations"),
        (5.23, 2.15, 1.55, 0.8, "Models +\nstrategies"),
        (7.08, 2.15, 1.30, 0.8, "AUROC +\nother metrics"),
        (8.68, 2.15, 1.40, 0.8, "Pre-specified\nscreen rule"),
        (10.38, 2.15, 1.35, 0.8, "Conclusion\nmatrix"),
    ]
    for spec in boxes:
        box(*spec)
    for left, right in zip(boxes, boxes[1:]):
        arrow(left[0] + left[2] + 0.04, 2.55, right[0] - 0.06, 2.55)
    ax.text(4.25, 1.82, "T1 / T2 tasks", ha="center", fontsize=9, color="#364453")
    ax.text(6.0, 1.82, "B0-B2 baselines", ha="center", fontsize=9, color="#364453")
    box(5.35, 4.0, 2.8, 0.75, "E8: unlabeled MSM\npretrain, then 10% labels", "#F0EAF6")
    box(7.1, 0.25, 2.8, 0.75, "E7: Chapman external\nvalidation (frozen models)", "#EBF3F8")
    arrow(6.0, 3.0, 6.6, 3.94, COLORS["vit"])
    arrow(8.12, 4.0, 8.05, 3.02, COLORS["vit"])
    arrow(6.75, 2.1, 7.8, 1.08, COLORS["resnet1d"])
    arrow(8.5, 1.04, 7.75, 2.08, COLORS["resnet1d"])
    ax.set_title("Study design: diagnosis, triage, and two extension branches", pad=8)
    save_figure(fig, out, "figA_study_design")


def fig_b(out: Path, runs: pd.DataFrame) -> None:
    d = full_t1(runs)
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    for j, model in enumerate(("resnet1d", "bilstm", "vit", "laresnet")):
        strategy = "S3_lead_agnostic" if model == "laresnet" else "S1_scratch"
        sub = d[(d.variant == model) & (d.strategy == strategy)]
        agg = sub.groupby("leads").macro_auroc.agg(["mean", "std"]).reindex(LEADS)
        if agg.isna().all().all():
            raise ValueError(f"Fig B missing model: {model}")
        x = np.arange(len(LEADS)) + (j - 1.5) * 0.075
        ax.errorbar(x, agg["mean"], yerr=agg["std"].fillna(0), marker="o", capsize=3,
                    linewidth=1.7, color=COLORS[model], label=LABELS[model])
    b2 = d[d.variant == "b2-features"].set_index("leads").macro_auroc
    ax.axhline(float(b2.loc["12lead"]), linestyle=":", linewidth=1.7,
               color=COLORS["b2-features"], label="B2 12-lead reference")
    ax.scatter([LEADS.index("leadI")], [float(b2.loc["leadI"])], marker="s", s=42,
               color=COLORS["b2-features"], label="B2 lead-I result", zorder=5)
    ax.set_xticks(range(len(LEADS)), LEAD_LABELS)
    ax.set(xlabel="Lead configuration", ylabel="Test macro AUROC", ylim=(0.72, 0.95),
           title="Diagnostic performance across lead configurations")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, loc="lower left", frameon=False)
    fig.tight_layout()
    save_figure(fig, out, "figB_lead_ladder")


def fig_c(out: Path, runs: pd.DataFrame) -> None:
    d = full_t1(runs)
    rows, names, label_colors = [], [], []
    for model, strategy in (("resnet1d", "S1_scratch"), ("laresnet", "S3_lead_agnostic")):
        for lead in LEADS:
            sub = d[(d.variant == model) & (d.strategy == strategy) & (d.leads == lead)]
            if len(sub) != 3:
                raise ValueError(f"Fig C requires three seeds: {model}, {lead}")
            rows.append([sub[f"auroc_{c}"].mean() for c in CLASSES])
            names.append(f"{LABELS[model]} / {dict(zip(LEADS, LEAD_LABELS))[lead]}")
            label_colors.append(COLORS[model])
    values = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(8.4, 6.5))
    im = ax.imshow(values, vmin=0.72, vmax=0.98, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(names)), names)
    for tick, color in zip(ax.get_yticklabels(), label_colors):
        tick.set_color(color)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            rgba = im.cmap(im.norm(values[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center",
                    color="#152332" if luminance > 0.47 else "white", fontsize=9)
    ax.axhline(5.5, color="white", linewidth=2)
    fig.colorbar(im, ax=ax, label="Test AUROC")
    ax.set_title("Per-class AUROC: ResNet1d and LA-ResNet")
    fig.tight_layout()
    save_figure(fig, out, "figC_per_class_heatmap")


def fig_d(out: Path, summary: Path) -> None:
    source = read_csv(summary / "conclusion_long.csv")
    d = source[(source.task == "t1") & (source.approach == "bilstm [S1_scratch]")]
    labels = CLASSES + ["ABNORMAL (derived 1-P(NORM))"]
    if len(d) != 30 or d.duplicated(["class", "leads"]).any():
        raise ValueError("Fig D requires the 6 x 5 primary-approach matrix")
    verdicts = ["not_replaceable", "conditional", "screen"]
    colors = ["#DFE3E7", "#F6D690", "#A8D8BE"]
    grid = np.empty((6, 5), dtype=int)
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    for i, label in enumerate(labels):
        for j, lead in enumerate(REDUCED):
            row = d[(d["class"] == label) & (d.leads == lead)].iloc[0]
            grid[i, j] = verdicts.index(row.verdict)
    ax.imshow(grid, cmap=ListedColormap(colors), vmin=-0.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(5), LEAD_LABELS[1:])
    ax.set_yticks(range(6), CLASSES + ["Derived triage"])
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i, label in enumerate(labels):
        for j, lead in enumerate(REDUCED):
            row = d[(d["class"] == label) & (d.leads == lead)].iloc[0]
            ax.text(j, i, f"{row.delta_auroc:+.3f}\n{row.sens_at_spec90:.2f}",
                    ha="center", va="center", fontsize=9, color="#152332")
    ax.set_title("Can reduced leads screen as well as the 12-lead reference?")
    fig.legend([Patch(facecolor=c, edgecolor="#88939C") for c in colors],
               ["Not replaceable", "Conditional", "Screen"], loc="lower center",
               bbox_to_anchor=(0.5, 0.06), ncol=3, frameon=False)
    fig.text(0.5, -0.005, "Cells: ΔAUROC (new − 12 leads) / sensitivity at 90% specificity. "
             "STTC–chest6 is screen in all five evaluated approaches.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    # The caption is checked against all five approaches, not hand-assumed.
    sttc = source[(source.task == "t1") & (source["class"] == "STTC") &
                  (source.leads == "chest6")]
    if len(sttc) != 5 or not (sttc.verdict == "screen").all():
        raise ValueError("Fig D STTC-chest6 caption disagrees with conclusion_long.csv")
    save_figure(fig, out, "figD_conclusion_matrix")


def fig_e(out: Path, runs: pd.DataFrame) -> None:
    d = runs[(runs.task == "t1") & runs.leads.isin(["12lead", "leadI"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, lead in zip(axes, ("12lead", "leadI")):
        for model, strategy in (("resnet1d", "S1_scratch"), ("laresnet", "S3_lead_agnostic"),
                                ("vit", "S1_scratch")):
            sub = d[(d.variant == model) & (d.strategy == strategy) & (d.leads == lead)]
            curve = sub.groupby("train_fraction").macro_auroc.mean().sort_index()
            ax.plot(curve.index * 100, curve.values, marker="o", color=COLORS[model],
                    linestyle="--" if model == "vit" else "-",
                    linewidth=1.8, label=LABELS[model])
        if lead == "leadI":
            transfer = d[(d.variant == "resnet1d-tfft") & (d.leads == lead)]
            curve = transfer.groupby("train_fraction").macro_auroc.mean().sort_index()
            ax.plot(curve.index * 100, curve.values, marker="s", linestyle=":",
                    color=COLORS["resnet1d"], label="ResNet1d transfer")
        msm = d[(d.variant == "vit-msm") & (d.leads == lead)]
        if len(msm) != 3:
            raise ValueError(f"Fig E missing three MSM seeds: {lead}")
        ax.scatter([10], [msm.macro_auroc.mean()], marker="D", s=85,
                   color=COLORS["vit"], edgecolor="white", linewidth=1,
                   zorder=6, label="ECG-ViT + MSM (3 seeds)")
        ax.set(xscale="log", xlabel="Labeled training data",
               title="12 leads" if lead == "12lead" else "Lead I")
        ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Test macro AUROC")
    fig.suptitle("Label efficiency and masked-signal pretraining")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.07),
               ncol=3, frameon=False)
    fig.text(0.5, 0.015, "ECG-ViT 10%: mean of 3 seeds; other E5 points: seed 0. "
             "Dashed ViT line spans unmeasured 25% and 50% fractions.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    save_figure(fig, out, "figE_label_efficiency")


def fig_f(out: Path, summary: Path) -> None:
    maps = {
        "v1": read_csv(summary / "external_chapman_v1_mapping/e7_external_t2.csv"),
        "v2": read_csv(summary / "external_chapman/e7_external_t2.csv"),
        "v2_lvhv": read_csv(summary / "external_chapman_v2_lvhv/e7_external_t2.csv"),
    }
    for key in maps:
        maps[key] = maps[key][(maps[key].source_task == "t2") &
                              maps[key].model.isin(["resnet1d", "laresnet"])]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex="col")
    xx = np.arange(len(LEADS))
    for model in ("resnet1d", "laresnet"):
        d = maps["v2"][maps["v2"].model == model].set_index("leads").reindex(LEADS)
        x = np.array([i for i, lead in enumerate(LEADS) if lead in maps["v2"][maps["v2"].model == model].leads.values])
        internal = d.iloc[x]
        for ax, col in ((axes[0, 0], "internal_ptb_test_auroc"),
                        (axes[1, 0], "internal_ptb_test_sens_at_spec90"),
                        (axes[0, 1], "external_auroc"),
                        (axes[1, 1], "external_sens_at_ptb_threshold")):
            ax.plot(x, internal[col], marker="o", linewidth=1.8,
                    linestyle="none" if model == "resnet1d" else "-",
                    color=COLORS[model], label=LABELS[model])
        for label, marker in (("v1", "x"), ("v2_lvhv", "s")):
            other = maps[label][maps[label].model == model].set_index("leads").reindex(LEADS).iloc[x]
            scatter_style = ({"facecolor": "none", "edgecolor": "#888888"} if marker == "s"
                             else {"color": "#888888"})
            axes[0, 1].scatter(x, other.external_auroc, marker=marker, s=44,
                               label=label if model == "resnet1d" else None,
                               zorder=5, **scatter_style)
        axes[0, 1].errorbar(x, internal.external_auroc,
                            yerr=[internal.external_auroc - internal.external_ci_low,
                                  internal.external_ci_high - internal.external_auroc],
                            fmt="none", ecolor=COLORS[model], capsize=2, alpha=0.8)
        axes[1, 1].scatter(x, internal.external_spec_at_ptb_threshold, marker="x", s=40,
                           color=COLORS[model], alpha=0.55)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
        ax.set_xticks(xx, LEAD_LABELS, rotation=15)
    axes[0, 0].set_title("Internal: PTB-XL test fold")
    axes[0, 1].set_title("External: Chapman v2 main mapping")
    axes[0, 0].set_ylabel("T2 AUROC")
    axes[0, 1].set_ylabel("T2 AUROC")
    axes[1, 0].set_ylabel("Sensitivity at 90% specificity")
    axes[1, 1].set_ylabel("Sensitivity / specificity at PTB threshold")
    axes[1, 0].axhline(0.8, color="#777777", linestyle=":")
    axes[1, 1].axhline(0.8, color="#777777", linestyle=":")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False, ncol=2)
    axes[1, 1].legend(handles=[
        Line2D([0], [0], color="#364453", marker="o", linestyle="none", label="Sensitivity"),
        Line2D([0], [0], color="#364453", marker="x", linestyle="none", label="Specificity"),
    ], frameon=False, loc="upper right")
    fig.text(0.5, 0.015, "Grey external AUROC markers: v1 (cross) and v2 + LVHV (square). "
             "ResNet1d has only 12-lead and lead-I points. Lower right uses thresholds fixed on PTB-XL validation.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_figure(fig, out, "figF_triage_external")


def fig_g(out: Path, runs: pd.DataFrame) -> None:
    d = full_t1(runs)
    cols = ["noise_20dB", "noise_10dB", "noise_5dB", "noise_0dB"]
    labels = ["clean", "20 dB", "10 dB", "5 dB", "0 dB"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lead in zip(axes, ("12lead", "leadI")):
        for model in ("resnet1d", "bilstm", "vit"):
            sub = d[(d.variant == model) & (d.strategy == "S1_scratch") & (d.leads == lead)]
            vals = [sub.macro_auroc.mean()] + [sub[col].mean() for col in cols]
            ax.plot(range(5), vals, color=COLORS[model], marker="o", linewidth=1.8,
                    label=LABELS[model])
        ax.set_xticks(range(5), labels)
        ax.set(xlabel="Added white noise (SNR)",
               title="12 leads" if lead == "12lead" else "Lead I")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Test macro AUROC")
    axes[0].legend(frameon=False)
    fig.suptitle("Noise robustness of scratch models")
    fig.tight_layout()
    save_figure(fig, out, "figG_noise_robustness")


def fig_h(out: Path, results: Path) -> None:
    grad = Image.open(results / "resnet1d_12lead_frac1_seed0/gradcam.png")
    attention = Image.open(results / "vit_12lead_frac1_seed0/attention_matrix.png")
    recon = Image.open(results / "vit-msm-pretrain_12lead_seed0/reconstruction.png")
    # The original Grad-CAM image has five equally spaced class panels in
    # NORM/MI/STTC/CD/HYP order; crop only the requested MI and STTC panels.
    w, h = grad.size
    if (w, h) != (1200, 1275):
        raise ValueError(f"Grad-CAM source dimensions changed: {grad.size}")
    mi = grad.crop((62, 292, 1185, 455))
    sttc = grad.crop((62, 534, 1185, 699))
    # Keep only the heatmap pixels; re-create its axes at the common 9 pt size.
    if attention.size != (900, 750):
        raise ValueError(f"attention source dimensions changed: {attention.size}")
    attention = attention.crop((95, 55, 710, 665))
    recon = recon.crop((0, 55, recon.width, recon.height))
    fig = plt.figure(figsize=(13, 9.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.35], width_ratios=[1.55, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[:2, 1]), fig.add_subplot(gs[2, :])]
    for ax, im, title in zip(axes, (mi, sttc, attention, recon),
                             ("A  ResNet1d Grad-CAM: MI", "B  ResNet1d Grad-CAM: STTC",
                              "C  ECG-ViT last-block attention: STTC", "D  MSM reconstruction: lead II")):
        if ax in axes[:2]:
            ax.imshow(im, extent=(0, 10, 0, 1), aspect="auto")
        else:
            ax.imshow(im)
        ax.set_title(title, loc="left", pad=5)
        if ax in axes[:2]:
            ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.set_yticks([])
            ax.set(xlabel="Time (s)", ylabel="Lead II")
        elif ax is axes[2]:
            ax.set_xticks(np.linspace(0, im.width - 1, 6), [str(x) for x in range(0, 51, 10)])
            ax.set_yticks(np.linspace(0, im.height - 1, 6), [str(x) for x in range(50, -1, -10)])
            ax.set(xlabel="Key token (0 = CLS)", ylabel="Query token (0 = CLS)")
        else:
            ax.axis("off")
    fig.text(0.5, 0.01, "Illustrative records only; Grad-CAM and attention are not causal clinical explanations.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, out, "figH_explainability")


def table_1(out: Path, results: Path, summary: Path) -> None:
    prevalence = pd.read_csv(results / "eda/table1_label_prevalence.csv", header=[0, 1], index_col=0)
    status = read_csv(summary / "external_chapman/record_status.csv")
    external = read_csv(summary / "external_chapman/e7_external_t2.csv").iloc[0]
    rows = []
    for split in ("train", "val", "test"):
        n = round(float(prevalence.loc["NORM", (split, "sum")]) /
                  float(prevalence.loc["NORM", (split, "mean")]))
        rows.append({"cohort": f"PTB-XL {split}", "measure": "all records", "count": n,
                     "denominator": n, "fraction": 1.0})
        for cls in CLASSES:
            count = int(prevalence.loc[cls, (split, "sum")])
            fraction = float(prevalence.loc[cls, (split, "mean")])
            if abs(count / n - fraction) > 1e-8:
                raise ValueError(f"Table 1 prevalence denominator mismatch: {split}, {cls}")
            rows.append({"cohort": f"PTB-XL {split}", "measure": cls, "count": count,
                         "denominator": n, "fraction": fraction})
    total = len(status)
    rows.append({"cohort": "Chapman v2", "measure": "all scanned", "count": total,
                 "denominator": total, "fraction": 1.0})
    for name, count in status.status.value_counts().items():
        readable = {
            "included": "included after signal checks",
            "excluded_unmapped_dx": "excluded: diagnosis outside target",
            "excluded_signal_flatline_lead": "excluded: flatline lead",
            "excluded_signal_nonfinite_signal": "excluded: nonfinite signal",
            "excluded_signal_time_data_'/'_does_not_match_format_'%d/%m/%Y'":
                "excluded: header date parse error",
        }.get(name)
        if readable is None:
            raise ValueError(f"Table 1 has an unmapped record status: {name}")
        rows.append({"cohort": "Chapman v2", "measure": readable, "count": int(count),
                     "denominator": total, "fraction": count / total})
    for name, count in (("included positive", int(external.external_positive)),
                        ("included negative", int(external.external_n - external.external_positive))):
        rows.append({"cohort": "Chapman v2", "measure": name, "count": count,
                     "denominator": total, "fraction": count / total})
    save_table(pd.DataFrame(rows), out, "table1_data_cohorts",
               "Sources: results/eda/table1_label_prevalence.csv and "
               "results/summary/external_chapman/{record_status,e7_external_t2}.csv. "
               "PTB-XL label rows are multi-label prevalence; Chapman fractions use all scanned records.")


def ensemble_ci(runs: pd.DataFrame, model: str) -> tuple[float, float, float]:
    d = runs[(runs.task == "t1") & (runs.variant == model) & (runs.strategy == "S1_scratch") &
             (runs.leads == "12lead") & (runs.train_fraction == 1.0)].sort_values("seed")
    if len(d) != 3:
        raise ValueError(f"Table 2 requires three full-data seeds: {model}")
    ys, ids, probs = [], [], []
    for row in d.itertuples():
        with np.load(Path(row.dir) / "preds.npz") as prediction:
            ys.append(prediction["test_y"])
            ids.append(prediction["test_ids"])
            probs.append(sigmoid(prediction["test_logits"]))
    if not all(np.array_equal(ys[0], y) for y in ys[1:]) or \
            not all(np.array_equal(ids[0], record_ids) for record_ids in ids[1:]):
        raise ValueError(f"Table 2 test samples are not aligned: {model}")
    average = np.mean(probs, axis=0)
    auc = macro_auroc(ys[0], average)
    lo, hi = bootstrap_ci(ys[0], average, macro_auroc, n=1000, seed=0)
    return auc, lo, hi


def table_2(out: Path, runs: pd.DataFrame, summary: Path) -> None:
    means = pd.read_csv(summary / "table_mean_std.csv", header=[0, 1], index_col=[0, 1, 2, 3, 4])
    ensemble = read_csv(summary / "seed_ensemble.csv")
    rows = []
    for model, strategy in (("b0-prior", "B_baseline"), ("b1-logreg", "B_baseline"),
                            ("b2-features", "B_baseline"), ("resnet1d", "S1_scratch"),
                            ("bilstm", "S1_scratch"), ("vit", "S1_scratch")):
        r = means.loc[("t1", model, strategy, "12lead", 1.0)]
        row = {"model": LABELS.get(model, model.upper()), "n_seeds": int(r[("macro_auroc", "count")])}
        for metric in ("macro_auroc", "macro_auprc", "macro_sens_at_spec90", "ece"):
            row[f"{metric}_mean"] = float(r[(metric, "mean")])
            row[f"{metric}_sd"] = float(r[(metric, "std")]) if pd.notna(r[(metric, "std")]) else np.nan
        if model in ("resnet1d", "bilstm", "vit"):
            reported = ensemble[(ensemble.task == "t1") & (ensemble.variant == model) &
                                (ensemble.leads == "12lead") & (ensemble.strategy == strategy)].iloc[0]
            auc, lo, hi = ensemble_ci(runs, model)
            if abs(auc - reported.ensemble_auroc) > 1e-8:
                raise ValueError(f"Table 2 recomputed ensemble disagrees with seed_ensemble.csv: {model}")
            row.update(ensemble_auroc=auc, ensemble_ci_low=lo, ensemble_ci_high=hi)
        else:
            row.update(ensemble_auroc=np.nan, ensemble_ci_low=np.nan, ensemble_ci_high=np.nan)
        rows.append(row)
    note = ("Sources: results/summary/table_mean_std.csv, seed_ensemble.csv, and aligned "
            "three-seed test predictions in results/<run>/preds.npz. Bootstrap CI: 1,000 record resamples, seed 0. "
            "Single-seed baselines have no across-seed SD or ensemble CI.")
    frame = pd.DataFrame(rows)
    save_table(frame, out, "table2_12lead_models", note)
    pretty = []
    for row in rows:
        record = {"Model": row["model"], "Seeds": row["n_seeds"]}
        for metric, label in (("macro_auroc", "AUROC"), ("macro_auprc", "AUPRC"),
                              ("macro_sens_at_spec90", "Sens@spec90"), ("ece", "ECE")):
            mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
            record[label] = f"{mean:.4f} ± {sd:.4f}" if np.isfinite(sd) else f"{mean:.4f} (1 seed)"
        record["Ensemble AUROC [95% CI]"] = (
            f"{row['ensemble_auroc']:.4f} [{row['ensemble_ci_low']:.4f}, {row['ensemble_ci_high']:.4f}]"
            if np.isfinite(row["ensemble_auroc"]) else "—")
        pretty.append(record)
    (out / "table2_12lead_models.md").write_text(
        "# Table 2: 12-lead models\n\n" + note + "\n\n" +
        pd.DataFrame(pretty).to_markdown(index=False) + "\n", encoding="utf-8")


def table_3(out: Path, summary: Path) -> None:
    all_tests = read_csv(summary / "paired_bootstrap_tests.csv")
    if len(all_tests) != 29:
        raise ValueError("Table 3 expects 29 comparisons in the Holm correction family")
    d = all_tests[all_tests.comparison.str.contains(
        "tfft|tffrz|laresnet", case=False, regex=True)].copy()
    note = ("Source: strategy rows of results/summary/paired_bootstrap_tests.csv. "
            "Differences are paired seed-ensemble AUROC comparisons. "
            "Holm adjustment covers all 29 comparisons in the source file, "
            "although only strategy-related rows are shown here. "
            "Values stored as zero after 1,000 bootstraps are displayed as < 0.001.")
    save_table(d, out, "table3_strategy_paired_tests", note)
    shown = d.copy()
    for col in ("p_value", "p_holm"):
        shown[col] = shown[col].map(
            lambda value: "< 0.001" if value == 0 else f"{value:.3f}")
    for col in ("diff", "ci95_lo", "ci95_hi"):
        shown[col] = shown[col].map(lambda value: f"{value:.4f}")
    shown.to_csv(out / "table3_strategy_paired_tests.csv", index=False)
    (out / "table3_strategy_paired_tests.md").write_text(
        "# Table 3: strategy paired tests\n\n" + note + "\n\n" +
        shown.to_markdown(index=False) + "\n", encoding="utf-8")


def table_4(out: Path, summary: Path) -> None:
    chunks = []
    for mapping, directory in (("v1", "external_chapman_v1_mapping"),
                               ("v2", "external_chapman"),
                               ("v2_lvhv", "external_chapman_v2_lvhv")):
        d = read_csv(summary / directory / "e7_external_t2.csv")
        d.insert(0, "mapping", mapping)
        chunks.append(d)
    frame = pd.concat(chunks, ignore_index=True)
    frame = frame[(frame.source_task == "t2") &
                  (((frame.model == "resnet1d") & frame.leads.isin(["12lead", "leadI"])) |
                   ((frame.model == "laresnet") & frame.leads.isin(["12lead", "leadI"])))].copy()
    frame = frame[["mapping", "model", "leads", "external_n", "external_positive",
                   "external_auroc", "external_ci_low", "external_ci_high",
                   "external_sens_at_spec90", "external_sens_at_ptb_threshold",
                   "external_spec_at_ptb_threshold", "internal_ptb_test_auroc",
                   "delta_auroc_external_minus_internal"]]
    save_table(frame, out, "table4_external_validation",
               "Sources: the three full-cohort e7_external_t2.csv files, restricted to T2 ResNet1d "
               "and LA-ResNet at 12 leads and lead I. v2 is the main mapping; v1 and v2_lvhv are "
               "sensitivity analyses. PTB-XL validation thresholds are transported unchanged.")


def table_5(out: Path, summary: Path) -> None:
    d = read_csv(summary / "e8_self_supervised.csv")
    paired = read_csv(summary / "e8_paired_tests.csv")
    rows = []
    for lead in ("12lead", "leadI"):
        by_method = d[d.leads == lead].set_index("method")
        comparison = paired[paired.leads == lead]
        if set(by_method.index) != {"scratch", "msm"} or len(comparison) != 1:
            raise ValueError(f"Table 5 requires scratch, MSM and one paired test for {lead}")
        scratch, msm = by_method.loc["scratch"], by_method.loc["msm"]
        test = comparison.iloc[0]
        if abs((msm.ensemble_auroc - scratch.ensemble_auroc) - test["diff"]) > 1e-8:
            raise ValueError(f"Table 5 paired difference disagrees with ensemble values: {lead}")
        if abs(msm.full_data_vit_reference_auroc - scratch.full_data_vit_reference_auroc) > 1e-8:
            raise ValueError(f"Table 5 full-data reference differs by method: {lead}")
        def method_text(row: pd.Series) -> str:
            return (f"{row.seed_mean_auroc:.4f} ± {row.seed_std_auroc:.4f}; "
                    f"{row.ensemble_auroc:.4f} [{row.ensemble_ci_low:.4f}, "
                    f"{row.ensemble_ci_high:.4f}]")
        rows.append({
            "leads": lead,
            "scratch (mean ± SD; ensemble [95% CI])": method_text(scratch),
            "MSM (mean ± SD; ensemble [95% CI])": method_text(msm),
            "MSM - scratch [95% CI]":
                f"{test['diff']:+.4f} [{test.ci95_low:+.4f}, {test.ci95_high:+.4f}]",
            "p": "< 0.001" if test.p_value == 0 else f"{test.p_value:.3f}",
            "full-data ViT reference": f"{scratch.full_data_vit_reference_auroc:.4f}",
        })
    save_table(pd.DataFrame(rows), out, "table5_self_supervised",
               "Sources: results/summary/e8_self_supervised.csv and e8_paired_tests.csv. "
               "Each row is one lead configuration; mean ± SD describes three seeds, followed by "
               "seed-ensemble AUROC [95% CI]. Paired difference is MSM minus scratch on "
               "seed-ensemble predictions with 1,000 record bootstraps. "
               "Full-data ViT is a reference, not part of the paired test.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/summary/paper_figs"))
    args = ap.parse_args()
    results = args.results.resolve()
    summary = results / "summary"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs = read_csv(summary / "all_runs.csv")
    fig_a(out)
    fig_b(out, runs)
    fig_c(out, runs)
    fig_d(out, summary)
    fig_e(out, runs)
    fig_f(out, summary)
    fig_g(out, runs)
    fig_h(out, results)
    table_1(out, results, summary)
    table_2(out, runs, summary)
    table_3(out, summary)
    table_4(out, summary)
    table_5(out, summary)
    print(f"Created 8 PNG + 8 PDF figures and 5 CSV + 5 Markdown tables in {out}")


if __name__ == "__main__":
    main()
