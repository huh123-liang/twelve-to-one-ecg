"""Reload a finished run (any model / strategy / task) together with its exact input arrays."""
from __future__ import annotations

import json
import os

from .data import TASK_CLASSES, Standardizer, apply_lead_mask, lead_indices, load_split, prepare_arrays, task_labels
from .models import build_model, load_weights


def load_run(run_dir: str, device="cpu"):
    """Returns (model, inputs, labels, splits, cfg). inputs/labels are dicts over train/val/test."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    ckpt_dir = os.path.join(os.path.dirname(run_dir.rstrip("/\\")), cfg["checkpoint_run"]) if cfg.get("checkpoint_run") else run_dir
    task = cfg.get("task", "t1")
    n_cls = len(TASK_CLASSES[task])
    splits = load_split(cfg["cache"], cfg["sr"], cfg.get("train_fraction", 1.0), cfg["seed"])
    labels = {k: task_labels(v.y5, task) for k, v in splits.items()}
    if cfg["model"] == "laresnet":
        std = Standardizer(splits["train"].X)
        inputs = {k: apply_lead_mask(std(v.X), cfg["leads"]) for k, v in splits.items()}
        n_in = 24
    else:
        inputs, _ = prepare_arrays(splits, cfg["leads"])
        n_in = len(lead_indices(cfg["leads"]))
    model = build_model(cfg["model"], n_in, n_cls, kernel=cfg.get("kernel", 7),
                        residual=not cfg.get("no_residual", False), cell=cfg.get("cell", "lstm"))
    load_weights(model, os.path.join(ckpt_dir, "best.pt"), map_location=device)
    return model.to(device).eval(), inputs, labels, splits, cfg
