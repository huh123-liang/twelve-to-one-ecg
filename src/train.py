"""Train and evaluate one configuration.

Examples
--------
python -m src.train --model resnet1d --leads 12lead --seed 0                     # S1 from scratch (T1)
python -m src.train --model vit --leads leadI --seed 1                           # ECG-ViT on lead I
python -m src.train --model resnet1d --leads 12lead --task t2                    # T2 triage
python -m src.train --model resnet1d --leads leadI --init_from results/resnet1d_12lead_frac1_seed0 --transfer finetune
python -m src.train --model laresnet --seed 0                                    # S3: one model, tested on all 6 lead sets
python -m src.train --model resnet1d --kernel 31 / --no_residual / --pos_weight  # ablations
python -m src.train --model bilstm --cell rnn                                    # vanilla RNN ablation

Training-loop structure follows Lecture 4: `run_epoch` (one pass, trains if an optimizer is given)
and `train_model` (early stopping on validation AUROC, keeping a deep copy of the best state_dict).
Curves are also written to TensorBoard (Lecture 7): `tensorboard --logdir results`.

Outputs (results/<run>/): config.json, history.csv, best.pt, preds.npz, metrics.json, model_summary.txt
For LA-ResNet the checkpoint lives in results/laresnet_all_.../ and every lead set gets its own
results/laresnet_<leads>_.../ folder with metrics.json + preds.npz, so summaries treat it like any run.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import (LEAD_SETS, TASK_CLASSES, ECGDataset, LeadAgnosticDataset, Standardizer, add_noise_snr,
                   apply_lead_mask, lead_indices, load_split, prepare_arrays, task_labels)
from .metrics import full_report, macro_auroc, sigmoid
from .models import build_model, count_params, load_weights, replace_input_layer
from .utils import get_device, set_seed


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet1d", choices=["resnet1d", "bilstm", "vit", "laresnet"])
    ap.add_argument("--leads", default="12lead", help="ignored for laresnet (trained on random subsets)")
    ap.add_argument("--task", default="t1", choices=["t1", "t2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--train_fraction", type=float, default=1.0)
    # ablations
    ap.add_argument("--kernel", type=int, default=7, help="ResNet block kernel size (receptive-field sweep)")
    ap.add_argument("--no_residual", action="store_true", help="plain CNN: remove skip connections")
    ap.add_argument("--cell", default="lstm", choices=["lstm", "rnn"], help="recurrent cell of CNN-BiLSTM")
    ap.add_argument("--pos_weight", action="store_true", help="BCEWithLogitsLoss(pos_weight=neg/pos)")
    # transfer learning (Lecture 8)
    ap.add_argument("--init_from", default="", help="run folder of a trained 12-lead resnet1d")
    ap.add_argument("--msm_from", default="", help="same-lead ECG-ViT masked-signal pretraining folder")
    ap.add_argument("--transfer", default="finetune", choices=["finetune", "frozen"])
    ap.add_argument("--backbone_lr_mult", type=float, default=0.1, help="discriminative LR for the backbone")
    # misc
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--sr", type=int, default=100)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0, help="keep 0 on Windows")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_tensorboard", action="store_true")
    ap.add_argument("--snr", type=float, nargs="*", default=[20, 10, 5, 0],
                    help="test-time noise levels (dB) for the robustness experiment")
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------
# naming
# ----------------------------------------------------------------------------

def variant_name(args) -> str:
    if args.msm_from:
        return "vit-msm"
    v = args.model
    if args.no_residual:
        v += "-plain"
    if args.kernel != 7:
        v += f"-k{args.kernel}"
    if args.cell != "lstm":
        v += f"-{args.cell}"
    if args.pos_weight:
        v += "-pw"
    if args.init_from:
        v += "-tf" + ("frz" if args.transfer == "frozen" else "ft")
    return v


def strategy_name(args) -> str:
    if args.msm_from:
        return "S4_self_supervised"
    if args.model == "laresnet":
        return "S3_lead_agnostic"
    if args.init_from:
        return "S2_transfer_" + args.transfer
    return "S1_scratch"


def run_tag(args, leads: str) -> str:
    task = "" if args.task == "t1" else f"_{args.task}"
    return f"{variant_name(args)}_{leads}{task}_frac{args.train_fraction:g}_seed{args.seed}"


# ----------------------------------------------------------------------------
# Lecture-4-style training loop
# ----------------------------------------------------------------------------

def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None, grad_clip=1.0,
              use_amp=False):
    """One pass over `loader`: trains if an optimizer is given, otherwise just evaluates."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss, n_seen = 0.0, 0
    all_logits, all_targets = [], []
    with torch.set_grad_enabled(is_training):
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)              # 1. clear old gradients
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            if is_training:
                scaler.scale(loss).backward()                      # 2. compute new gradients
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # keep updates bounded (Lecture 4)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)                             # 3. update the weights
                scaler.update()
                # AMP can skip an optimizer step on overflow; keep the LR schedule aligned (Lecture 4).
                if scheduler is not None and scaler.get_scale() >= scale_before:
                    scheduler.step()
            total_loss += loss.item() * len(xb)
            n_seen += len(xb)
            if not is_training:
                all_logits.append(logits.float().cpu())
                all_targets.append(yb.cpu())
    out = {"loss": total_loss / max(n_seen, 1)}
    if not is_training:
        out["logits"] = torch.cat(all_logits).numpy()
        out["auroc"] = macro_auroc(torch.cat(all_targets).numpy(), sigmoid(out["logits"]))
    return out


@torch.no_grad()
def predict(model, X, device, batch=256):
    """X: (N, T, C) numpy -> logits (N, K)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch].transpose(0, 2, 1))).to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out)


def train_model(model, train_loader, val_fn, criterion, optimizer, scheduler, device, args, out_dir, use_amp):
    """Early stopping on validation macro AUROC; restores the best weights (Lecture 4)."""
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
        except Exception:
            writer = None
    best_auc, best_state, bad, hist = -np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler, use_amp=use_amp)
        val_loss, val_auc = val_fn()
        hist.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": val_loss, "val_macro_auroc": val_auc,
                     "lr": scheduler.get_last_lr()[0], "time_s": time.time() - t0})
        if writer is not None:
            writer.add_scalar("Loss/train", tr["loss"], epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("AUROC/validation", val_auc, epoch)
            writer.flush()
        print(f"  epoch {epoch:3d} | loss {tr['loss']:.4f} | val_loss {val_loss:.4f} val_auroc {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, bad = val_auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {epoch}")
                break
    if writer is not None:
        writer.close()
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "best.pt"))
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    return best_auc, hist


def write_summary(model, n_ch, seq_len, out_dir):
    """Layer-by-layer summary with torchinfo (Lecture 3), without touching the RNG or BN statistics."""
    try:
        from torchinfo import summary
        model.eval()
        with torch.random.fork_rng(devices=[]):
            s = summary(model, input_data=torch.zeros(1, n_ch, seq_len, device=next(model.parameters()).device),
                        verbose=0, depth=3)
        with open(os.path.join(out_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
            f.write(str(s))
    except Exception as e:  # torchinfo is optional
        with open(os.path.join(out_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"torchinfo unavailable: {e}\nparams: {count_params(model)}\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def run(args):
    if args.msm_from and (args.model != "vit" or args.init_from):
        raise ValueError("--msm_from requires --model vit and cannot be combined with --init_from")
    msm_dir = args.msm_from
    if msm_dir:
        with open(os.path.join(msm_dir, "config.json"), encoding="utf-8") as f:
            msm_cfg = json.load(f)
        if msm_cfg.get("model") != "vit-msm-pretrain" or msm_cfg.get("leads") != args.leads:
            raise ValueError("--msm_from must contain ECG-ViT pretraining for the same lead set")
        if not os.path.isfile(os.path.join(msm_dir, "encoder.pt")):
            raise FileNotFoundError("--msm_from has no completed encoder.pt")
    set_seed(args.seed)
    device = get_device(args.device)
    is_la = args.model == "laresnet"
    if is_la and args.init_from:
        raise ValueError("LA-ResNet is trained from scratch")
    main_leads = "all" if is_la else args.leads
    tag = run_tag(args, main_leads)
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)
    cfg = vars(args).copy()
    cfg.update({"variant": variant_name(args), "strategy": strategy_name(args), "run": tag})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    class_names = TASK_CLASSES[args.task]
    splits = load_split(args.cache, args.sr, args.train_fraction, args.seed)
    y = {k: task_labels(v.y5, args.task) for k, v in splits.items()}

    if is_la:
        std = Standardizer(splits["train"].X)
        arrs12 = {k: std(v.X) for k, v in splits.items()}
        train_ds = LeadAgnosticDataset(arrs12["train"], y["train"], augment=True)
        val_sets = {l: apply_lead_mask(arrs12["val"], l) for l in LEAD_SETS}
        n_in = 24
    else:
        arrs, std = prepare_arrays(splits, args.leads)
        train_ds = ECGDataset(arrs["train"], y["train"], augment=True)
        n_in = len(lead_indices(args.leads))
    print(f"[{tag}] device={device} train={len(y['train'])} val={len(y['val'])} test={len(y['test'])}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              drop_last=True, pin_memory=device.type == "cuda")

    # ---- model ----
    if args.init_from:
        src_cfg = json.load(open(os.path.join(args.init_from, "config.json")))
        if src_cfg["model"] != "resnet1d" or src_cfg.get("leads") != "12lead":
            raise ValueError("--init_from must point to a 12-lead resnet1d run")
        model = build_model("resnet1d", 12, len(TASK_CLASSES[src_cfg.get("task", "t1")]))
        load_weights(model, os.path.join(args.init_from, "best.pt"))
        model = replace_input_layer(model, n_in, keep_channels=lead_indices(args.leads))
        if len(class_names) != model.head[-1].out_features:
            model.head[-1] = nn.Linear(model.head[-1].in_features, len(class_names))
    else:
        model = build_model(args.model, n_in, len(class_names), kernel=args.kernel,
                            residual=not args.no_residual, cell=args.cell)
        if msm_dir:
            # Lecture 8: initialize the shared encoder, but learn a fresh supervised head.
            encoder_state = torch.load(os.path.join(msm_dir, "encoder.pt"), map_location="cpu", weights_only=True)
            encoder_state = {k: v for k, v in encoder_state.items() if not k.startswith("mlp_head.")}
            missing, unexpected = model.load_state_dict(encoder_state, strict=False)
            if unexpected or not missing or any(not k.startswith("mlp_head.") for k in missing):
                raise RuntimeError(f"MSM encoder mismatch: missing={missing}, unexpected={unexpected}")
    model = model.to(device)

    # ---- optimiser: plain AdamW, or frozen backbone / discriminative LR for transfer (Lecture 8) ----
    new_params = list(model.stem[0].parameters()) + list(model.head.parameters()) if args.init_from else []
    new_ids = {id(p) for p in new_params}
    if args.init_from and args.transfer == "frozen":
        for p in model.parameters():
            p.requires_grad = id(p) in new_ids        # train only the new input layer and the classifier
        groups = [{"params": new_params, "lr": args.lr}]
    elif args.init_from:
        backbone = [p for p in model.parameters() if id(p) not in new_ids]
        groups = [{"params": new_params, "lr": args.lr},
                  {"params": backbone, "lr": args.lr * args.backbone_lr_mult}]
    else:
        groups = [{"params": model.parameters(), "lr": args.lr}]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in optimizer.param_groups],
                                                    total_steps=steps, pct_start=0.1)

    pos_weight = None
    if args.pos_weight:   # class imbalance (Lecture 4): weight positives by neg/pos of the TRAIN set
        pos = y["train"].sum(0)
        pos_weight = torch.tensor(np.maximum(1.0, (len(y["train"]) - pos) / np.maximum(pos, 1)), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    plain_bce = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda" and not args.no_amp

    # ---- validation function used for early stopping ----
    if is_la:
        def val_fn():   # one model must work for every lead set: early-stop on the mean over all six
            aucs, losses = [], []
            for l, Xv in val_sets.items():
                lg = predict(model, Xv, device)
                aucs.append(macro_auroc(y["val"], sigmoid(lg)))
                losses.append(float(plain_bce(torch.from_numpy(lg), torch.from_numpy(y["val"]))))
            return float(np.mean(losses)), float(np.mean(aucs))
    else:
        # Validation goes through `predict` rather than a DataLoader: iterating a DataLoader draws a
        # seed from the global RNG, which would silently change the training augmentation sequence.
        def val_fn():
            lg = predict(model, arrs["val"], device)
            loss = float(plain_bce(torch.from_numpy(lg), torch.from_numpy(y["val"])))
            return loss, macro_auroc(y["val"], sigmoid(lg))

    n_params = count_params(model)
    best, hist = train_model(model, train_loader, val_fn, criterion, optimizer, scheduler, device, args, out_dir,
                             use_amp)
    write_summary(model, n_in, splits["train"].X.shape[1], out_dir)

    # ---- final evaluation with the best weights ----
    eval_sets = list(LEAD_SETS) if is_la else [args.leads]
    rng = np.random.default_rng(args.seed)
    reports = {}
    for l in eval_sets:
        if is_la:
            Xv, Xt = val_sets[l], apply_lead_mask(arrs12["test"], l)
            raw_test = splits["test"].X
            to_input = lambda raw, l=l: apply_lead_mask(std(raw), l)
            ev_dir = os.path.join(args.out, run_tag(args, l))
            os.makedirs(ev_dir, exist_ok=True)
            with open(os.path.join(ev_dir, "config.json"), "w") as f:
                json.dump({**cfg, "leads": l, "checkpoint_run": tag}, f, indent=2)
        else:
            Xv, Xt = arrs["val"], arrs["test"]
            raw_test = splits["test"].X[:, :, lead_indices(l)]
            to_input = std
            ev_dir = out_dir
        val_logits, test_logits = predict(model, Xv, device), predict(model, Xt, device)
        rep = full_report(y["test"], test_logits, y["val"], val_logits, class_names)
        rep.update({"run": os.path.basename(ev_dir), "model": args.model, "variant": variant_name(args),
                    "strategy": strategy_name(args), "task": args.task, "leads": l, "seed": args.seed,
                    "train_fraction": args.train_fraction, "n_train": int(len(y["train"])), "n_params": n_params,
                    "best_val_macro_auroc": best, "epochs_run": len(hist),
                    "train_time_s": hist[-1]["time_s"] if hist else 0.0, "checkpoint_run": tag})
        rep["noise_robustness"] = {}
        for snr in args.snr:   # robustness: additive white noise on the raw test signal
            noisy = to_input(add_noise_snr(raw_test, snr, rng))
            rep["noise_robustness"][f"{snr:g}dB"] = macro_auroc(y["test"], sigmoid(predict(model, noisy, device)))
        np.savez_compressed(os.path.join(ev_dir, "preds.npz"), val_logits=val_logits, val_y=y["val"],
                            test_logits=test_logits, test_y=y["test"], test_ids=splits["test"].ids,
                            test_age=splits["test"].age, test_sex=splits["test"].sex, test_y5=splits["test"].y5)
        with open(os.path.join(ev_dir, "metrics.json"), "w") as f:
            json.dump(rep, f, indent=2)
        reports[l] = rep
        ci = rep["macro_auroc_ci95"]
        print(f"[{rep['run']}] TEST macro AUROC {rep['macro_auroc']:.4f} (95% CI {ci[0]:.4f}-{ci[1]:.4f}) "
              f"AUPRC {rep['macro_auprc']:.4f} sens@spec90 {rep['macro_sens_at_spec90']:.4f} ECE {rep['ece']:.4f}")
    return reports if is_la else reports[args.leads]


if __name__ == "__main__":
    run(parse_args())
