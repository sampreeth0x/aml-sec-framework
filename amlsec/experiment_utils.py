"""Shared experiment plumbing: bundles, seeds, result persistence."""
import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def save_result(name: str, payload: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[results] wrote {path}", flush=True)


def new_bundle(seed=0, train_n_per_class=2000, test_n_per_class=400):
    from amlsec.data import DatasetBundle
    return DatasetBundle(train_n_per_class=train_n_per_class,
                         test_n_per_class=test_n_per_class, seed=seed)


def run_stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")