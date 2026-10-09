"""Step 1: build the signal cache from the raw PTB-XL download.

python scripts/prepare_data.py --ptbxl data/ptb-xl --cache data/cache
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import CLASSES, build_cache, load_metadata  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ptbxl", default="data/ptb-xl", help="folder containing ptbxl_database.csv")
ap.add_argument("--cache", default="data/cache")
ap.add_argument("--sr", type=int, default=100, choices=[100, 500])
a = ap.parse_args()

if not os.path.exists(os.path.join(a.ptbxl, "ptbxl_database.csv")):
    sys.exit(f"ptbxl_database.csv not found in {a.ptbxl}. See README section 'Download PTB-XL'.")

db = load_metadata(a.ptbxl)
print(f"records: {len(db)}  patients: {db['patient_id'].nunique()}  without superclass: {(db['n_labels'] == 0).sum()}")
print("label prevalence:", {c: int(db[c].sum()) for c in CLASSES})
path = build_cache(a.ptbxl, a.cache, a.sr)
print("cache written:", path)
