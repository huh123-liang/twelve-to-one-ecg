"""Create a small fake dataset in the exact PTB-XL file layout.

Only for testing the pipeline when the real data is not available yet.
Signals are synthetic sine/QRS-like waves whose shape depends on the label,
so models can learn something. NEVER report results obtained on this data.

python scripts/make_mock_ptbxl.py --out data/mock-ptb-xl --n 600
"""
import argparse
import os

import numpy as np
import pandas as pd
import wfdb

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CODE_OF = {"NORM": "NORM", "MI": "IMI", "STTC": "NDT", "CD": "CLBBB", "HYP": "LVH"}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/mock-ptb-xl")
ap.add_argument("--n", type=int, default=600)
a = ap.parse_args()
rng = np.random.default_rng(0)

scp = pd.DataFrame({"diagnostic": [1.0] * 5 + [np.nan],
                    "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", np.nan]},
                   index=list(CODE_OF.values()) + ["SR"])
os.makedirs(a.out, exist_ok=True)
scp.to_csv(os.path.join(a.out, "scp_statements.csv"))

rows = []
t = np.arange(1000) / 100.0
for i in range(1, a.n + 1):
    labs = ["NORM"] if rng.random() < 0.45 else list(rng.choice(["MI", "STTC", "CD", "HYP"], size=rng.integers(1, 3), replace=False))
    hr = rng.uniform(55, 100) / 60
    base = np.zeros((1000, 12))
    for k in range(12):
        phase = (t * hr) % 1.0
        qrs = np.exp(-((phase - 0.3) ** 2) / (0.0004 if "CD" not in labs else 0.0015))
        twave = (0.3 if "STTC" not in labs else -0.2) * np.exp(-((phase - 0.6) ** 2) / 0.004)
        amp = (1.0 + 0.6 * ("HYP" in labs)) * (1 + 0.1 * k)
        qwave = -0.3 * ("MI" in labs) * np.exp(-((phase - 0.27) ** 2) / 0.0002)
        base[:, k] = amp * qrs + twave + qwave + 0.05 * rng.standard_normal(1000)
    sub = f"{(i // 1000) * 1000:05d}"
    rel = f"records100/{sub}/{i:05d}_lr"
    os.makedirs(os.path.join(a.out, "records100", sub), exist_ok=True)
    wfdb.wrsamp(f"{i:05d}_lr", fs=100, units=["mV"] * 12, sig_name=LEADS, p_signal=base.astype(np.float64),
                fmt=["16"] * 12, write_dir=os.path.join(a.out, "records100", sub))
    rows.append({"ecg_id": i, "patient_id": float(1 + i // 2), "age": float(rng.integers(20, 90)), "sex": int(rng.integers(0, 2)),
                 "scp_codes": str({CODE_OF[l]: 100.0 for l in labs} | {"SR": 0.0}),
                 "strat_fold": int(1 + (i // 2) % 10), "filename_lr": rel, "filename_hr": rel.replace("records100", "records500")})
pd.DataFrame(rows).set_index("ecg_id").to_csv(os.path.join(a.out, "ptbxl_database.csv"))
print(f"mock PTB-XL with {a.n} records written to {a.out}")
