"""Explainability figures for a trained run (Lectures 6 and 8).

* Grad-CAM: for each class, the most confident true positive in the test set, with the
  class-specific saliency overlaid on one lead. Works for ResNet1d, CNN-BiLSTM, ECG-ViT
  (Grad-CAM for Transformer on the patch tokens) and LA-ResNet.
* ECG-ViT only: attention of the CLS token to each 200 ms patch in the last block
  (which parts of the recording the Transformer "reads" when deciding).

python scripts/explain.py --run results/resnet1d_12lead_frac1_seed0
python scripts/explain.py --run results/vit_12lead_frac1_seed0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data import LEAD_SETS, LEADS, TASK_CLASSES  # noqa: E402
from src.gradcam import GradCAM1d  # noqa: E402
from src.inference import load_run  # noqa: E402
from src.metrics import sigmoid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--lead", default="II", help="lead to draw (falls back to the first available lead)")
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
a = ap.parse_args()

if a.device == "cuda" and not torch.cuda.is_available():
    sys.exit("CUDA requested but unavailable")
model, inputs, labels, splits, cfg = load_run(a.run, device=a.device)
print(f"explain device={a.device} run={a.run}")
classes = TASK_CLASSES[cfg.get("task", "t1")]
present = LEAD_SETS[cfg["leads"]]
# column of the lead to draw inside the model input
if cfg["model"] == "laresnet":
    lead = a.lead if a.lead in present else present[0]
    col = LEADS.index(lead)
else:
    lead = a.lead if a.lead in present else present[0]
    col = present.index(lead)

X = inputs["test"]
y = labels["test"]
with torch.no_grad():
    logits = []
    for start in range(0, len(X), 32):
        x_batch = torch.from_numpy(np.ascontiguousarray(X[start:start + 32].transpose(0, 2, 1))).to(a.device)
        logits.append(model(x_batch).cpu().numpy())
    p = sigmoid(np.concatenate(logits, axis=0))
fs = cfg["sr"]
cam = GradCAM1d(model)
is_vit = cfg["model"] == "vit"

rows = len(classes)
fig, axes = plt.subplots(rows, 2 if is_vit else 1, figsize=(11 if is_vit else 8, 1.7 * rows), squeeze=False)
for k, c in enumerate(classes):
    pos = np.where(y[:, k] == 1)[0]
    ax = axes[k, 0]
    if not len(pos):
        ax.set_title(f"{c}: no positive test example")
        continue
    i = pos[np.argmax(p[pos, k])]
    x = torch.from_numpy(np.ascontiguousarray(X[i].T[None])).to(a.device)
    s = cam(x, k)
    sig = X[i][:, col]
    t = np.arange(len(sig)) / fs
    ax.imshow(s[None], extent=[t[0], t[-1], sig.min(), sig.max()], aspect="auto", cmap="Reds", alpha=0.45, zorder=1)
    ax.plot(t, sig, color="k", lw=0.7, zorder=3)
    ax.set_title(f"Grad-CAM · {c} (ecg_id {splits['test'].ids[i]}, p={p[i, k]:.2f}, lead {lead})", fontsize=8)
    if is_vit:
        with torch.no_grad():
            model(x)
        att = model.cls_attention()[0].cpu().numpy()
        att = np.repeat(att / att.max(), model.patch_size)
        ax2 = axes[k, 1]
        ax2.imshow(att[None], extent=[t[0], t[-1], sig.min(), sig.max()], aspect="auto", cmap="Blues", alpha=0.5, zorder=1)
        ax2.plot(t, sig, color="k", lw=0.7, zorder=3)
        ax2.set_title(f"CLS attention (last block) · {c}", fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("time (s)")
fig.tight_layout()
out = os.path.join(a.run, "gradcam.png")
fig.savefig(out, dpi=150)
print("saved", out)

if is_vit:
    # Lecture 6/7: show the full last-block token-to-token attention matrix.
    k = list(classes).index("STTC") if "STTC" in classes else 0
    pos = np.where(y[:, k] == 1)[0]
    if len(pos):
        i = pos[np.argmax(p[pos, k])]
        x = torch.from_numpy(np.ascontiguousarray(X[i].T[None])).to(a.device)
        with torch.no_grad():
            model(x)
        weights = model.blocks[-1].attn.last_weights[0].mean(0).cpu().numpy()
        fig_att, ax_att = plt.subplots(figsize=(6, 5))
        im = ax_att.imshow(weights, origin="lower", aspect="auto", cmap="Blues")
        ax_att.set(xlabel="Key token (0 = CLS)", ylabel="Query token (0 = CLS)",
                   title=f"Last-block attention · {classes[k]} (ecg_id {splits['test'].ids[i]})")
        fig_att.colorbar(im, ax=ax_att, label="Mean attention over heads")
        fig_att.tight_layout()
        out_att = os.path.join(a.run, "attention_matrix.png")
        fig_att.savefig(out_att, dpi=150)
        plt.close(fig_att)
        print("saved", out_att)
