"""Figure A (study design), journal style. Pure matplotlib drawing; no results are read.

python scripts/make_figA_study_design.py --out results/summary/paper_figs
Writes figA_study_design.pdf (vector) and figA_study_design.png (300 dpi).
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

NAVY, TEAL, RUST = "#1F3A5F", "#2F6F73", "#A23B2A"
INK, MUTED, RULE = "#1F2A37", "#5B6675", "#C3CAD3"
GREY_BG, TEAL_BG, RUST_BG, WHITE = "#F4F5F7", "#EAF2F2", "#F6EAE7", "#FFFFFF"
SAND = "#E9D8A6"

families = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Inter", "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"] if f in families), "sans-serif")
plt.rcParams.update({"font.family": FONT, "font.size": 6.5, "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def box(ax, x, y, w, h, fc=WHITE, ec=RULE, lw=0.6, r=0.6, ls="-", z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def text(ax, x, y, s, size=6.5, color=INK, weight="normal", ha="left", va="center", style="normal", z=5):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, style=style, zorder=z)


def arrow(ax, p0, p1, color=MUTED, lw=0.8, ls="-", rad=0.0, z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def ecg(t):
    """Synthetic PQRST complex train (illustration only)."""
    y = np.zeros_like(t)
    for c in np.arange(0.35, 4.0, 0.9):
        y += 0.12 * np.exp(-((t - c + 0.18) / 0.035) ** 2)          # P
        y += -0.10 * np.exp(-((t - c - 0.025) / 0.012) ** 2)        # Q
        y += 1.00 * np.exp(-((t - c) / 0.014) ** 2)                 # R
        y += -0.22 * np.exp(-((t - c + 0.03) / 0.012) ** 2)         # S
        y += 0.25 * np.exp(-((t - c - 0.28) / 0.07) ** 2)           # T
    return y


def bracket(ax, x0, x1, y, label, sub):
    ax.plot([x0, x0, x1, x1], [y - 0.7, y, y, y - 0.7], color=MUTED, lw=0.6, zorder=3)
    ax.plot([(x0 + x1) / 2] * 2, [y, y + 0.7], color=MUTED, lw=0.6, zorder=3)
    ax.text((x0 + x1) / 2, y + 2.0, label, fontsize=6.6, color=NAVY, fontweight="bold", ha="right", va="center")
    ax.text((x0 + x1) / 2, y + 2.0, "  ·  " + sub, fontsize=6.2, color=MUTED, ha="left", va="center")


def panel_title(ax, x, y, letter, title):
    text(ax, x, y, letter, size=8, weight="bold")
    text(ax, x + 2.2, y, title, size=6.8, weight="bold", color=NAVY)


def main(out):
    fig = plt.figure(figsize=(7.2, 3.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    bracket(ax, 1, 38, 46.0, "Stage 1", "data and inputs")
    bracket(ax, 40.5, 69.5, 46.0, "Stage 2", "102 runs, 3 seeds")
    bracket(ax, 72.5, 99, 46.0, "Stage 3", "test fold")

    # ---------------- (a) PTB-XL
    panel_title(ax, 1, 42.3, "a", "PTB-XL v1.0.3")
    box(ax, 1, 21.0, 17.5, 19.0, fc=WHITE, ec=NAVY, lw=0.8)
    t = np.linspace(0, 3.4, 600)
    for i, ld in enumerate(["I", "II", "V1", "V5"]):
        yb = 36.6 - i * 4.0
        sig = ecg(t + 0.12 * i) * (-0.7 if ld == "V1" else 0.9)
        ax.plot(3.8 + t * 4.1, yb + sig * 1.8, color=INK, lw=0.45, zorder=4)
        text(ax, 1.9, yb, ld, size=5.4, color=MUTED)
    text(ax, 9.75, 22.6, "12 leads · 10 s · 100 Hz", size=5.4, color=MUTED, ha="center")
    text(ax, 1, 18.2, "21,388 labelled ECGs", size=6.2, weight="bold", color=INK)
    text(ax, 1, 16.2, "18,869 patients", size=5.6, color=MUTED)
    y0, h = 10.6, 2.6
    for x, w, c in [(1.0, 12.0, "#DDE3EA"), (13.0, 2.75, TEAL), (15.75, 2.75, NAVY)]:
        ax.add_patch(Rectangle((x, y0), w, h, fc=c, ec=WHITE, lw=0.6, zorder=3))
    text(ax, 7.0, y0 + h / 2, "train", size=5.6, color=INK, ha="center")
    text(ax, 1, 8.6, "official folds 1–8 / 9 / 10", size=5.4, color=MUTED)
    text(ax, 1, 6.9, "17,084 / 2,146 / 2,158", size=5.4, color=INK)
    text(ax, 14.4, 14.6, "val", size=5.0, color=TEAL, ha="center")
    text(ax, 17.1, 14.6, "test", size=5.0, color=NAVY, ha="center")

    # ---------------- (b) lead configurations
    panel_title(ax, 21, 42.3, "b", "Six lead sets")
    cfg = [("12 leads", range(12)), ("limb 6", range(6)), ("chest 6", range(6, 12)),
           ("I + II", [0, 1]), ("lead II", [1]), ("lead I", [0])]
    gx, gy, cw, ch = 28.4, 36.0, 0.8, 2.2
    for lab, a, b in [("limb", 0, 6), ("chest", 6, 12)]:
        x0, x1 = gx + a * cw, gx + b * cw - 0.15
        ax.plot([x0, x0, x1, x1], [gy + ch + 0.6, gy + ch + 1.0, gy + ch + 1.0, gy + ch + 0.6], color=MUTED, lw=0.5)
        text(ax, (x0 + x1) / 2, gy + ch + 2.0, lab, size=5.0, color=MUTED, ha="center")
    for i, (lab, on) in enumerate(cfg):
        yy = gy - i * (ch + 0.8)
        watch = lab == "lead I"
        text(ax, 21.2, yy + ch / 2, lab, size=5.8, color=RUST if watch else INK, weight="bold" if watch else "normal")
        for j in range(12):
            fc = (RUST if watch else NAVY) if j in on else "#E7EAEE"
            ax.add_patch(Rectangle((gx + j * cw, yy), cw - 0.15, ch, fc=fc, ec="none", zorder=3))
    text(ax, 21.2, 17.6, "lead I = smartwatch-like", size=5.4, color=RUST)
    box(ax, 21, 11.9, 17.0, 4.0, fc=GREY_BG, ec="none", r=0.5)
    text(ax, 21.8, 14.6, "T1 diagnosis", size=5.8, weight="bold", color=NAVY)
    text(ax, 21.8, 12.9, "NORM · MI · STTC · CD · HYP", size=5.0, color=INK)
    box(ax, 21, 7.2, 17.0, 4.0, fc=GREY_BG, ec="none", r=0.5)
    text(ax, 21.8, 9.9, "T2 triage", size=5.8, weight="bold", color=NAVY)
    text(ax, 21.8, 8.2, "normal vs any abnormality", size=5.0, color=INK)
    arrow(ax, (18.7, 30.5), (20.6, 30.5))

    # ---------------- (c) models and strategies
    panel_title(ax, 40.5, 42.3, "c", "Models and strategies")
    box(ax, 40.5, 7.2, 29.0, 33.0, fc=WHITE, ec=TEAL, lw=0.9, r=0.9)
    arch = [("ResNet1d", "4.1 M · residual conv", "res"), ("CNN-BiLSTM", "0.8 M · conv + BiLSTM", "lstm"),
            ("ECG-ViT", "0.8 M · patch Transformer", "vit")]
    for i, (nm, sub, kind) in enumerate(arch):
        yy = 33.6 - i * 6.2
        box(ax, 42.0, yy, 26.0, 5.4, fc=GREY_BG, ec="none", r=0.5)
        text(ax, 43.0, yy + 3.7, nm, size=6.2, weight="bold", color=NAVY)
        text(ax, 43.0, yy + 1.6, sub, size=5.2, color=MUTED)
        ix = 60.0
        if kind == "res":
            for k in range(5):
                hh = 3.6 - k * 0.5
                ax.add_patch(Rectangle((ix + k * 1.5, yy + 2.7 - hh / 2), 1.05, hh, fc=NAVY, ec="none", zorder=3))
        elif kind == "lstm":
            for k in range(5):
                ax.add_patch(Rectangle((ix + k * 1.5, yy + 1.0), 1.05, 1.2, fc=TEAL, ec="none", zorder=3))
                ax.add_patch(Rectangle((ix + k * 1.5, yy + 3.0), 1.05, 1.2, fc=TEAL_BG, ec=TEAL, lw=0.4, zorder=3))
                if k < 4:
                    ax.plot([ix + k * 1.5 + 1.05, ix + (k + 1) * 1.5], [yy + 3.6] * 2, color=TEAL, lw=0.5, zorder=3)
        else:
            for k in range(5):
                ax.add_patch(Rectangle((ix + k * 1.5, yy + 0.9), 1.05, 1.05, fc=RUST if k == 0 else "#C9D2DC",
                                       ec="none", zorder=3))
            for a in range(5):
                for b in range(5):
                    if a != b:
                        ax.plot([ix + a * 1.5 + 0.52, ix + b * 1.5 + 0.52], [yy + 2.0, yy + 4.5], color=NAVY,
                                lw=0.2, alpha=0.45, zorder=2)
    text(ax, 42.0, 19.6, "+ baselines B0–B2 (prior, logistic regression)", size=5.2, color=MUTED)
    strat = [("S1", "scratch"), ("S2", "transfer"), ("S3", "LA-ResNet"), ("S4", "MSM")]
    for i, (code, lab) in enumerate(strat):
        xx = 42.0 + i * 6.6
        ours = code == "S3"
        box(ax, xx, 9.0, 6.0, 8.2, fc=RUST_BG if ours else TEAL_BG, ec="none", r=0.5)
        text(ax, xx + 3.0, 15.0, code, size=6.6, weight="bold", color=RUST if ours else TEAL, ha="center")
        text(ax, xx + 3.0, 12.4, lab, size=5.2, color=INK, ha="center")
    text(ax, 48.6 + 6.6 + 3.0, 10.4, "ours", size=4.8, color=RUST, ha="center", style="italic")
    text(ax, 45.0, 10.4, "", size=4.8)
    text(ax, 45.0 + 6.6, 10.4, "12 → k leads", size=4.6, color=MUTED, ha="center")
    text(ax, 45.0 + 19.8, 10.4, "self-sup.", size=4.6, color=MUTED, ha="center")
    arrow(ax, (38.3, 30.5), (40.3, 30.5))

    # ---------------- (d) evaluation
    panel_title(ax, 72.5, 42.3, "d", "Statistics and verdict")
    box(ax, 72.5, 33.2, 26.5, 7.0, fc=GREY_BG, ec="none")
    text(ax, 73.5, 38.4, "AUROC · AUPRC · sens@spec90 · ECE", size=5.6, weight="bold", color=INK)
    text(ax, 73.5, 36.3, "3-seed ensemble · paired bootstrap", size=5.3, color=INK)
    text(ax, 73.5, 34.5, "Holm-corrected · tuned on validation", size=5.3, color=INK)
    arrow(ax, (69.7, 36.7), (72.3, 36.7))
    box(ax, 72.5, 22.4, 26.5, 8.8, fc=WHITE, ec=RUST, lw=0.8, ls="--")
    text(ax, 73.5, 29.4, "Screen rule, fixed in advance", size=5.8, weight="bold", color=RUST)
    ax.text(73.5, 26.6, r"$\mathrm{CI}_{95}^{\,\mathrm{low}}(\Delta\mathrm{AUROC}) \geq {-}0.02$", fontsize=6.4, color=INK,
            va="center")
    ax.text(73.5, 24.0, r"and  $\mathrm{sens@spec}_{90} \geq 0.80$", fontsize=6.4, color=INK, va="center")
    arrow(ax, (85.7, 33.0), (85.7, 31.4))
    text(ax, 72.5, 19.6, "Conclusion matrix", size=6.0, weight="bold", color=NAVY)
    mx, my, s = 72.5, 8.0, 1.5
    for r in range(6):
        for c in range(5):
            fc = TEAL if (r == 2 and c == 1) else "#E7EAEE"
            ax.add_patch(Rectangle((mx + c * (s + 0.25), my + (5 - r) * 1.6), s, 1.35, fc=fc, ec="none", zorder=3))
    lx = 82.4
    for k, (c, lab) in enumerate([(TEAL, "screen"), (SAND, "conditional"), ("#E7EAEE", "not replaceable")]):
        ax.add_patch(Rectangle((lx, 15.6 - k * 2.6), 1.3, 1.3, fc=c, ec="none", zorder=3))
        text(ax, lx + 1.9, 16.25 - k * 2.6, lab, size=5.4, color=INK)
    text(ax, lx, 8.9, "6 rows × 5 lead sets", size=5.0, color=MUTED)
    arrow(ax, (85.7, 22.2), (85.7, 20.8))

    # ---------------- branches
    box(ax, 21.0, 0.8, 48.5, 4.6, fc=WHITE, ec=TEAL, lw=0.7, ls="--")
    text(ax, 22.0, 3.1, "E8", size=6.2, weight="bold", color=TEAL)
    text(ax, 25.2, 3.1, "masked-signal pre-training on unlabelled ECGs, then ECG-ViT with 10% of labels", size=5.2,
         color=INK)
    arrow(ax, (64.8, 5.6), (64.8, 8.8), color=TEAL)
    box(ax, 72.5, 0.8, 26.5, 4.6, fc=WHITE, ec=NAVY, lw=0.7, ls="--")
    text(ax, 73.5, 3.1, "E7", size=6.2, weight="bold", color=NAVY)
    text(ax, 76.6, 3.9, "Chapman, 34,815 external ECGs", size=5.2, color=INK)
    text(ax, 76.6, 2.2, "frozen T2 models, fixed threshold", size=5.2, color=MUTED)
    arrow(ax, (69.7, 12.5), (73.6, 5.6), color=NAVY, rad=-0.3)

    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "figA_study_design.pdf"))
    fig.savefig(os.path.join(out, "figA_study_design.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/summary/paper_figs")
    main(ap.parse_args().out)
