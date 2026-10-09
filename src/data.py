"""PTB-XL data loading for multi-label superdiagnostic ECG classification.

Pipeline
--------
1. `build_cache()` reads PTB-XL metadata + WFDB records once and stores
   signals as a float32 .npy array (N, T, 12) together with labels/folds.
2. `load_split()` returns train/val/test arrays using the official
   stratified folds (1-8 train, 9 val, 10 test), which keep all records of
   one patient in the same fold (no patient leakage).
3. `ECGDataset` applies light training-time augmentation (Lecture 8: data augmentation).
   `LeadAgnosticDataset` additionally drops random leads and appends a lead mask
   (our LA-ResNet design; the mask idea mirrors the padding mask of Lectures 5-6).

Tasks
-----
t1 : 5 diagnostic superclasses, multi-label (one sigmoid per class)
t2 : normal vs abnormal triage (abnormal = any of MI/STTC/CD/HYP), one sigmoid
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# PTB-XL stores leads in this order.
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
TASK_CLASSES = {"t1": CLASSES, "t2": ["ABNORMAL"]}

# Lead configurations studied in the reduced-lead experiment.
LEAD_SETS = {
    "12lead": LEADS,
    "limb6": ["I", "II", "III", "AVR", "AVL", "AVF"],
    "chest6": ["V1", "V2", "V3", "V4", "V5", "V6"],
    "2lead": ["I", "II"],
    "leadII": ["II"],    # classic rhythm-strip lead
    "leadI": ["I"],      # smartwatch-style single lead
}
REDUCED_SETS = ["limb6", "chest6", "2lead", "leadII", "leadI"]


def lead_indices(lead_set: str) -> list[int]:
    if lead_set not in LEAD_SETS:
        raise ValueError(f"unknown lead set {lead_set!r}; choose from {list(LEAD_SETS)}")
    return [LEADS.index(l) for l in LEAD_SETS[lead_set]]


def task_labels(y5: np.ndarray, task: str) -> np.ndarray:
    """Turn the 5-class multi-hot matrix into the labels of the chosen task."""
    if task == "t1":
        return y5.astype(np.float32)
    if task == "t2":
        return (y5[:, 1:].sum(axis=1, keepdims=True) > 0).astype(np.float32)
    raise ValueError(f"unknown task {task!r}")


# ----------------------------------------------------------------------------
# Metadata and labels
# ----------------------------------------------------------------------------

def load_metadata(ptbxl_dir: str) -> pd.DataFrame:
    """Load ptbxl_database.csv and attach multi-hot superdiagnostic labels."""
    db = pd.read_csv(os.path.join(ptbxl_dir, "ptbxl_database.csv"), index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(os.path.join(ptbxl_dir, "scp_statements.csv"), index_col=0)
    scp = scp[scp["diagnostic"] == 1]

    def to_super(codes: dict) -> list[str]:
        # Following the PTB-XL reference code we keep every listed diagnostic
        # statement regardless of its likelihood value.
        out = {scp.loc[c, "diagnostic_class"] for c in codes if c in scp.index}
        return sorted(x for x in out if isinstance(x, str))

    db["superclasses"] = db["scp_codes"].apply(to_super)
    for c in CLASSES:
        db[c] = db["superclasses"].apply(lambda s, c=c: int(c in s))
    db["n_labels"] = db[CLASSES].sum(axis=1)
    return db


def build_cache(ptbxl_dir: str, cache_dir: str, sampling_rate: int = 100) -> str:
    """Read all WFDB records once and save a compact cache (.npy + .csv)."""
    import wfdb  # imported lazily so training does not require wfdb
    from tqdm import tqdm

    os.makedirs(cache_dir, exist_ok=True)
    db = load_metadata(ptbxl_dir)
    db = db[db["n_labels"] > 0]  # records without a diagnostic superclass are dropped
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"

    signals = []
    for fn in tqdm(db[col], desc=f"reading WFDB @ {sampling_rate} Hz"):
        sig, meta = wfdb.rdsamp(os.path.join(ptbxl_dir, fn))
        if [s.upper() for s in meta["sig_name"]] != LEADS:
            raise RuntimeError(f"unexpected lead order in {fn}: {meta['sig_name']}")
        signals.append(sig.astype(np.float32))
    X = np.stack(signals)  # (N, T, 12)
    X = np.nan_to_num(X)

    sig_path = os.path.join(cache_dir, f"signals_{sampling_rate}hz.npy")
    np.save(sig_path, X)
    keep = ["patient_id", "age", "sex", "strat_fold"] + CLASSES
    db[keep].to_csv(os.path.join(cache_dir, f"meta_{sampling_rate}hz.csv"))
    return sig_path


@dataclass
class Split:
    X: np.ndarray      # (N, T, 12)
    y5: np.ndarray     # (N, 5) superclass multi-hot
    ids: np.ndarray    # ecg_id
    age: np.ndarray
    sex: np.ndarray    # PTB-XL: 0 = male, 1 = female


def load_split(cache_dir: str, sampling_rate: int = 100, train_fraction: float = 1.0,
               seed: int = 0) -> dict[str, Split]:
    X = np.load(os.path.join(cache_dir, f"signals_{sampling_rate}hz.npy"), mmap_mode="r")
    meta = pd.read_csv(os.path.join(cache_dir, f"meta_{sampling_rate}hz.csv"), index_col=0)
    y = meta[CLASSES].values.astype(np.float32)
    folds = meta["strat_fold"].values
    ids = meta.index.values

    parts = {"train": folds <= 8, "val": folds == 9, "test": folds == 10}
    out = {}
    for name, mask in parts.items():
        idx = np.where(mask)[0]
        if name == "train" and train_fraction < 1.0:
            # Subsample by patient so the label-efficiency study does not leak (Lecture 8: how much data is enough).
            rng = np.random.default_rng(seed)
            pats = meta["patient_id"].values[idx]
            uniq = np.unique(pats)
            chosen = set(rng.choice(uniq, size=max(1, int(len(uniq) * train_fraction)), replace=False))
            idx = idx[np.array([p in chosen for p in pats])]
        out[name] = Split(np.asarray(X[idx]), y[idx], ids[idx], meta["age"].values[idx], meta["sex"].values[idx])
    return out


# ----------------------------------------------------------------------------
# Standardisation
# ----------------------------------------------------------------------------

class Standardizer:
    """Per-lead z-score fitted on the training set only (Lectures 1 and 4: fit the scaler on TRAIN only)."""

    def __init__(self, X_train: np.ndarray):
        self.mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = X_train.std(axis=(0, 1), keepdims=True).astype(np.float32) + 1e-6

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


def prepare_arrays(splits: dict[str, Split], lead_set: str):
    """Select leads, fit the standardiser on train and apply it to all splits."""
    li = lead_indices(lead_set)
    arrs = {k: v.X[:, :, li] for k, v in splits.items()}
    std = Standardizer(arrs["train"])
    return {k: std(a) for k, a in arrs.items()}, std


def add_noise_snr(X: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at a given SNR (per record and lead), for robustness tests."""
    p_sig = (X ** 2).mean(axis=1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return (X + rng.standard_normal(X.shape).astype(np.float32) * np.sqrt(p_noise)).astype(np.float32)


# ----------------------------------------------------------------------------
# Torch datasets
# ----------------------------------------------------------------------------

def augment_batchless(x: torch.Tensor) -> torch.Tensor:
    """Amplitude scaling, baseline offset, gaussian noise, circular time shift. x: (C, T)."""
    x = x * (1 + 0.1 * torch.randn(x.shape[0], 1))
    x = x + 0.05 * torch.randn(x.shape[0], 1)
    x = x + 0.02 * torch.randn_like(x)
    return torch.roll(x, shifts=int(torch.randint(-50, 51, (1,))), dims=1)


class ECGDataset(Dataset):
    """Channels-first (C, T) samples for Conv1d (Lecture 5: Conv1d wants channels first)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1)))
        self.y = torch.from_numpy(y)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = augment_batchless(x)
        return x, self.y[i]


def lead_mask_vector(lead_set: str) -> np.ndarray:
    m = np.zeros(len(LEADS), dtype=np.float32)
    m[lead_indices(lead_set)] = 1.0
    return m


def apply_lead_mask(X12: np.ndarray, lead_set: str) -> np.ndarray:
    """Zero the missing leads and append 12 constant mask channels -> (N, T, 24)."""
    m = lead_mask_vector(lead_set)
    Xm = X12 * m[None, None, :]
    mask_ch = np.broadcast_to(m[None, None, :], X12.shape)
    return np.concatenate([Xm, mask_ch], axis=2).astype(np.float32)


class LeadAgnosticDataset(Dataset):
    """Random lead-subset training for LA-ResNet.

    With probability p_full the full 12 leads are kept; otherwise one of the five
    reduced configurations (or, with probability p_random, a random subset) is used.
    Missing leads are zeroed (= the training mean after standardisation) and a
    0/1 mask tells the network which leads are real.
    """

    def __init__(self, X12: np.ndarray, y: np.ndarray, augment: bool = True, p_full: float = 0.5,
                 p_random: float = 0.2):
        self.X = torch.from_numpy(np.ascontiguousarray(X12.transpose(0, 2, 1)))  # (N, 12, T)
        self.y = torch.from_numpy(y)
        self.augment = augment
        self.p_full, self.p_random = p_full, p_random
        self.masks = [torch.from_numpy(lead_mask_vector(s)) for s in REDUCED_SETS]

    def __len__(self) -> int:
        return len(self.y)

    def _sample_mask(self) -> torch.Tensor:
        r = float(torch.rand(1))
        if r < self.p_full:
            return torch.ones(len(LEADS))
        if r < self.p_full + self.p_random:
            m = (torch.rand(len(LEADS)) < 0.5).float()
            if m.sum() == 0:
                m[int(torch.randint(0, len(LEADS), (1,)))] = 1.0
            return m
        return self.masks[int(torch.randint(0, len(self.masks), (1,)))]

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = augment_batchless(x)
        m = self._sample_mask()
        x = x * m[:, None]
        x = torch.cat([x, m[:, None].expand(-1, x.shape[1])], dim=0)  # (24, T)
        return x, self.y[i]
