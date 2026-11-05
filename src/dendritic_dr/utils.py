"""Shared helpers: paths, seeding, device selection, result logging."""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

# Repo root = two levels above this file (src/dendritic_dr/utils.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def results_root(smoke: bool = False) -> Path:
    """Real runs write to results/; offline smoke runs to results/smoke/ (git-ignored)."""
    return RESULTS_DIR / "smoke" if smoke else RESULTS_DIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_num_workers() -> int:
    """DataLoader workers: 0 on Windows (spawn-based multiprocessing is slow and
    fragile there), otherwise up to 4."""
    if os.name == "nt":
        return 0
    return min(4, os.cpu_count() or 1)


def append_result(csv_path: Path, row: dict) -> None:
    """Append one row to a CSV, writing the header if the file is new.
    Nested values (lists/dicts) are stored as JSON strings. If the row has columns
    the file does not have yet (e.g. a new option was added), the file is rewritten
    with the new columns appended to the header and left empty for older rows."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            writer.writeheader()
            writer.writerow(flat)
        return
    with csv_path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    missing = [k for k in flat if k not in header]
    if missing:
        with csv_path.open(newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
        header = header + missing
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header, restval="")
            writer.writeheader()
            writer.writerows(old_rows)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=header, restval="").writerow(flat)
