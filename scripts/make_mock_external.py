"""Build a small WFDB-format external library from mock PTB-XL test signals.

The signals are upsampled to 500 Hz to exercise the E7 downsampling code.
Extra AF-only and flat-lead records exercise the pre-specified exclusions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
import wfdb  # noqa: E402

from src.data import LEADS, load_split  # noqa: E402
from scripts.external_validation import MAPPING_V1  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/mock-cache")
    ap.add_argument("--out_dir", required=True, help="D-drive directory for the mock external library")
    return ap.parse_args(argv)


def write_record(parent: Path, name: str, signal100: np.ndarray, codes: list[str]):
    target = parent / "WFDBRecords" / "00" / "000"
    target.mkdir(parents=True, exist_ok=True)
    signal500 = resample_poly(signal100, 5, 1, axis=0)
    wfdb.wrsamp(name, fs=500, units=["mV"] * 12, sig_name=LEADS,
                p_signal=signal500.astype(np.float64), fmt=["16"] * 12,
                adc_gain=[1000] * 12, baseline=[0] * 12,
                comments=["Dx: " + ",".join(codes)],
                write_dir=str(target))
    # wrsamp chooses .dat for format 16; Chapman stores the same WFDB payload as .mat.
    (target / f"{name}.dat").rename(target / f"{name}.mat")
    header = target / f"{name}.hea"
    header.write_text(header.read_text(encoding="utf-8").replace(f"{name}.dat", f"{name}.mat"),
                      encoding="utf-8")


def run(args):
    root = Path(args.out_dir).resolve()
    if root.drive.upper() != "D:":
        raise ValueError("the E7 mock external library must be stored on D:")
    if (root / "WFDBRecords").exists() and list((root / "WFDBRecords").rglob("*.hea")):
        raise FileExistsError(f"refusing to overwrite existing WFDB records in {root / 'WFDBRecords'}")
    split = load_split(args.cache, 100, train_fraction=1.0, seed=0)["test"]
    code_for_class = {name: sorted(codes)[0] for name, codes in MAPPING_V1.items()}
    records = []
    for i, (signal, y5) in enumerate(zip(split.X, split.y5)):
        codes = [code_for_class[name] for name, positive in zip(("MI", "STTC", "CD", "HYP"), y5[1:]) if positive]
        if not codes:
            codes = ["426783006"]
        name = f"MOCK{i:05d}"
        write_record(root, name, signal, codes)
        records.append((name, ",".join(codes)))
    write_record(root, "MOCKAF000", split.X[0], ["164889003"])
    flat = split.X[1].copy()
    flat[:, 0] = 0
    write_record(root, "MOCKFLAT0", flat, [code_for_class["MI"]])
    names = pd.DataFrame([
        ("SR", "Sinus Rhythm", "426783006"),
        ("AFIB", "Atrial Fibrillation", "164889003"),
        ("MI", "myocardial infarction", code_for_class["MI"]),
        ("STTC", "ST-T Change", code_for_class["STTC"]),
        ("CD", "conduction disturbance", code_for_class["CD"]),
        ("HYP", "hypertrophy", code_for_class["HYP"]),
    ], columns=["Acronym Name", "Full Name", "Snomed_CT"])
    names.to_csv(root / "ConditionNames_SNOMED-CT.csv", index=False)
    print(f"Mock external library: {len(records)+2} WFDB records at {root / 'WFDBRecords'}")
    print("Deliberate exclusions: AF-only MOCKAF000 and flat-lead MOCKFLAT0")


if __name__ == "__main__":
    run(parse_args())
