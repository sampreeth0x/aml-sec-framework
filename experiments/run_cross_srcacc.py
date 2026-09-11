"""Supplementary measurement: source-class accuracy of each CLEAN-trained regime.

Needed for the corrected composite poisoning score:
    SourceDrop = Acc_clean(source class) - Acc_poisoned(source class)
i.e. source-class accuracy compared against source-class accuracy (same
population), not overall clean accuracy against source-class accuracy.

Train-only: no MIA / extraction / PGD, so this is cheap. Same regimes, seeds
and data pipeline as run_cross_threat.py (imports train_regime from it).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cross_threat import SEEDS, SOURCE, train_regime

from amlsec.experiment_utils import new_bundle, save_result
from amlsec.models import evaluate


def main():
    bundle = new_bundle()
    src_mask = bundle.test_y == SOURCE
    out = {}
    for regime in ["baseline", "auditfilter", "at", "dp", "ls"]:
        rows = []
        for seed in SEEDS:
            model, _ = train_regime(regime, bundle.train_raw_x,
                                    bundle.train_raw_y.copy(), seed)
            preds, _, _ = evaluate(model, bundle.test_x, bundle.test_y)
            rows.append({
                "seed": seed,
                "src_class_accuracy": float((preds[src_mask] == SOURCE).mean()),
                "clean_accuracy": float((preds == bundle.test_y).mean())})
            print(f"[srcacc] {regime} s={seed} src={rows[-1]['src_class_accuracy']:.4f} "
                  f"acc={rows[-1]['clean_accuracy']:.4f}", flush=True)
        out[regime] = rows
    save_result("cross_srcacc", out)
    print("[srcacc] DONE", flush=True)


if __name__ == "__main__":
    main()