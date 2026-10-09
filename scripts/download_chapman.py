"""Resumable, SHA256-verified Chapman WFDB download to D:.

All headers are retained to account for every exclusion. Signal .mat files are
downloaded only for records meeting the frozen E7 SNOMED label rule. This avoids
keeping a 2.3 GB ZIP beside the extracted records on a capacity-limited drive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests  # noqa: E402

from scripts.external_validation import MAPPINGS, map_labels, read_codes  # noqa: E402


DEFAULT_BASE = "https://physionet.org/files/ecg-arrhythmia/1.0.0/"
LOCAL = threading.local()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="D-drive dataset root; WFDBRecords is created underneath")
    ap.add_argument("--base_url", default=DEFAULT_BASE, help="official files URL or an exact mirror of version 1.0.0")
    ap.add_argument("--stage", choices=["headers", "signals", "all"], default="all")
    ap.add_argument("--mapping", choices=tuple(MAPPINGS), default="v2")
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args(argv)


def session():
    if not hasattr(LOCAL, "http"):
        LOCAL.http = requests.Session()
        LOCAL.http.headers["User-Agent"] = "BS6204-E7-checksummed-research-download/1.0"
    return LOCAL.http


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, target: Path, sha256: str | None = None):
    if target.is_file() and (sha256 is None or checksum(target) == sha256):
        return "verified_existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(4):
        try:
            with session().get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with partial.open("wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if sha256 is not None and checksum(partial) != sha256:
                raise IOError(f"SHA256 mismatch for {url}")
            os.replace(partial, target)
            return "downloaded_verified"
        except (requests.RequestException, OSError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"failed {url} after 4 attempts: {exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def manifest_entries(path: Path):
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"bad SHA256SUMS line: {line[:100]!r}")
        rel = parts[1].strip().lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"unsafe SHA256SUMS path: {rel}")
        entries[rel_path.as_posix()] = parts[0].lower()
    if len(entries) < 90000:
        raise ValueError(f"manifest has too few files: {len(entries)}")
    return entries


def download_many(paths, entries, dest, base_url, workers):
    paths = sorted(set(paths))
    print(f"checking/downloading {len(paths)} checksummed files", flush=True)
    complete = CounterLike()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), 500):
            batch = paths[start:start + 500]
            futures = {pool.submit(fetch, base_url + path, dest / path, entries[path]): path for path in batch}
            for future in as_completed(futures):
                complete.add(future.result())
            i = start + len(batch)
            free_gb = shutil.disk_usage(dest).free / 2**30
            print(f"{i}/{len(paths)} checked; downloaded={complete.downloaded}; "
                  f"already_verified={complete.existing}; D_free={free_gb:.2f} GiB", flush=True)
            if free_gb < 0.45:
                raise RuntimeError("D: free space below 0.45 GiB; download stopped for capacity")


class CounterLike:
    def __init__(self):
        self.downloaded = 0
        self.existing = 0

    def add(self, result):
        if result == "downloaded_verified":
            self.downloaded += 1
        else:
            self.existing += 1


def run(args):
    dest = Path(args.dest).resolve()
    if dest.drive.upper() != "D:":
        raise ValueError("the Chapman external library must be stored on D:")
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must be in [1, 32]")
    base = args.base_url.rstrip("/") + "/"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / "SHA256SUMS.txt"
    fetch(base + "SHA256SUMS.txt", manifest)
    if manifest.stat().st_size != 8711012:
        raise ValueError("official Chapman v1.0.0 SHA256SUMS.txt size mismatch")
    entries = manifest_entries(manifest)
    metadata = [path for path in entries if not path.endswith((".hea", ".mat"))]
    headers = [path for path in entries if path.endswith(".hea")]
    if args.stage in ("headers", "all"):
        download_many(metadata + headers, entries, dest, base, args.workers)
    if args.stage in ("signals", "all"):
        missing_headers = [path for path in headers if not (dest / path).is_file()]
        if missing_headers:
            raise RuntimeError(f"download headers before signals ({len(missing_headers)} missing)")
        selected = []
        for path in headers:
            triage, _, _ = map_labels(read_codes(dest / path), args.mapping)
            if triage is not None:
                signal_path = str(Path(path).with_suffix(".mat")).replace("\\", "/")
                if signal_path not in entries:
                    raise RuntimeError(f"no .mat checksum for {path}")
                selected.append(signal_path)
        print(f"{args.mapping} label map selects {len(selected)} of {len(headers)} signal files", flush=True)
        download_many(selected, entries, dest, base, args.workers)
    print(f"Chapman v1.0.0 download stage {args.stage} complete; all selected files verified against SHA256SUMS.txt")


if __name__ == "__main__":
    run(parse_args())
