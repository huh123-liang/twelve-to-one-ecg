"""Unlabeled masked-signal pretraining of ECG-ViT on PTB-XL folds 1-8 (Lecture 7).

The ninth fold is used only for reconstruction-loss early stopping. The tenth fold and
all diagnostic labels are unused. ``encoder.pt`` is written last as the completion marker.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data import ECGDataset, LEAD_SETS, load_split, prepare_arrays  # noqa: E402
from src.models import MaskedSignalModel  # noqa: E402
from src.utils import get_device, set_seed  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", required=True, choices=["12lead", "leadI"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_amp", action="store_true")
    return ap.parse_args(argv)


def validation_mse(model, X, device, batch, seed):
    """Use the same masks each epoch without advancing the training RNG."""
    model.eval()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    weighted_loss = weighted_zero = n_seen = 0.0
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(10000 + seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(10000 + seed)
        for start in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[start:start + batch].transpose(0, 2, 1))).to(device)
            loss, _, mask = model(xb)
            target = model.encoder.to_patches(xb)
            zero_mse = target.float().square()[mask].mean()
            weighted_loss += float(loss) * len(xb)
            weighted_zero += float(zero_mse) * len(xb)
            n_seen += len(xb)
    return weighted_loss / n_seen, weighted_zero / n_seen


def save_reconstruction(model, X, leads, seed, device, out_dir):
    model.eval()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(10000 + seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(10000 + seed)
        xb = torch.from_numpy(np.ascontiguousarray(X[:1].transpose(0, 2, 1))).to(device)
        _, pred, mask = model(xb)
    chosen_lead = "II" if leads == "12lead" else "I"
    col = LEAD_SETS[leads].index(chosen_lead)
    patch_size = model.encoder.patch_size
    signal = X[0, :, col]
    predicted = pred[0].float().cpu().numpy().reshape(-1, len(LEAD_SETS[leads]), patch_size)
    hidden = mask[0].cpu().numpy()
    t = np.arange(len(signal)) / 100
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(t, signal, color="black", lw=0.75, label="Original standardized ECG")
    for j in np.flatnonzero(hidden):
        segment = slice(j * patch_size, (j + 1) * patch_size)
        ax.axvspan(t[segment.start], t[segment.stop - 1] + 0.01, color="#f2bb77", alpha=0.35)
        ax.plot(t[segment], predicted[j, col], color="#bd4b2e", lw=1.2,
                label="Masked-patch reconstruction" if j == np.flatnonzero(hidden)[0] else None)
    ax.set(xlabel="Time (s)", ylabel="Standardized amplitude", title=f"MSM reconstruction, lead {chosen_lead}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reconstruction.png"), dpi=150)
    plt.close(fig)


def run(args):
    if args.seed != 0:
        raise ValueError("the pre-specified E8 pretraining seed is 0")
    if args.epochs < 1 or args.batch < 1:
        raise ValueError("epochs and batch must be positive")
    set_seed(args.seed)
    device = get_device(args.device)
    tag = f"vit-msm-pretrain_{args.leads}_seed{args.seed}"
    out_dir = os.path.join(args.out, tag)
    marker = os.path.join(out_dir, "encoder.pt")
    if os.path.exists(marker):
        print(f"skip {tag}: encoder.pt exists")
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    cfg = {"model": "vit-msm-pretrain", "leads": args.leads, "seed": args.seed,
           "labels_used": False, "mask_ratio": 0.15, "cache": args.cache, "sr": 100,
           "train_folds": "1-8", "validation_fold": 9, "epochs_max": args.epochs,
           "batch": args.batch, "lr": 1e-3, "weight_decay": 1e-2, "patience": 10}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    splits = load_split(args.cache, 100, train_fraction=1.0, seed=args.seed)
    arrs, _ = prepare_arrays(splits, args.leads)  # Lecture 1: scaler fits folds 1-8 only
    y_placeholder = np.zeros((len(arrs["train"]), 1), dtype=np.float32)
    train_ds = ECGDataset(arrs["train"], y_placeholder, augment=True)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=device.type == "cuda")
    model = MaskedSignalModel(n_leads=len(LEAD_SETS[args.leads]), mask_ratio=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                     total_steps=args.epochs * len(loader), pct_start=0.1)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[{tag}] device={device} train={len(train_ds)} val={len(arrs['val'])} masked_patches=8/50")
    start_time = time.time()
    best_mse, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = n_seen = 0.0
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, _, _ = model(xb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            loss_sum += float(loss.detach()) * len(xb)
            n_seen += len(xb)
        val_mse, zero_mse = validation_mse(model, arrs["val"], device, args.batch, args.seed)
        row = {"epoch": epoch, "train_mse": loss_sum / n_seen, "val_mse": val_mse,
               "zero_prediction_mse": zero_mse, "lr": scheduler.get_last_lr()[0],
               "time_s": time.time() - start_time}
        history.append(row)
        print(f"  epoch {epoch:3d} | train_mse {row['train_mse']:.5f} | val_mse {val_mse:.5f} | zero_mse {zero_mse:.5f}")
        if val_mse < best_mse:
            best_mse, best_state, bad = val_mse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 10:
                print(f"  early stop at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    zero_mse = validation_mse(model, arrs["val"], device, args.batch, args.seed)[1]
    summary = {"best_val_mse": best_mse, "zero_prediction_mse": zero_mse,
               "ratio_to_zero": best_mse / zero_mse, "best_epoch": int(np.argmin([r["val_mse"] for r in history]) + 1),
               "epochs_run": len(history), "train_time_s": time.time() - start_time}
    with open(os.path.join(out_dir, "pretrain_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_reconstruction(model, arrs["val"], args.leads, args.seed, device, out_dir)
    torch.save(model.encoder.state_dict(), marker)  # completion marker: write this last
    print(f"[{tag}] saved {marker}; best val MSE {best_mse:.5f}; ratio to zero {summary['ratio_to_zero']:.3f}")
    return out_dir


if __name__ == "__main__":
    run(parse_args())
