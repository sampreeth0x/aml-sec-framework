"""Experiment A — poisoning dose-response + detection-before-degradation.

For each poison fraction f in {0, 1, 3, 5, 10}% and each of 3 seeds:
  - flip f of class-2 training labels to class 7 (targeted, stealthy)
  - retrain the identical CNN
  - measure: clean test accuracy, per-class accuracy, ECE, confusion matrix,
    source-class accuracy (where the damage should concentrate), flip rate
  - run the kNN data audit on the poisoned dataset and record whether it can
    flag poisoned samples (recall@1% flagged) BEFORE retraining sees the damage
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from amlsec.attacks.poisoning import (poison_label_flips, knn_label_audit,
                                      chi_square_label_shift)
from amlsec.data import to_float
from amlsec.defenses import audit_and_filter
from amlsec.experiment_utils import save_result, run_stamp, new_bundle
from amlsec.metrics import accuracy, expected_calibration_error, binomial_ci
from amlsec.models import make_model, train_model, evaluate

SOURCE, TARGET = 2, 7
FRACTIONS = [0.0, 0.01, 0.03, 0.05, 0.10]
SEEDS = [0, 1, 2]
EPOCHS = 4


def main():
    bundle = new_bundle()
    clean_labels = bundle.train_raw_y.copy()
    rng = np.random.default_rng(0)
    clean_audit = knn_label_audit(bundle.train_raw_x, clean_labels, k=5)

    results = {"config": {"source": SOURCE, "target": TARGET, "fractions": FRACTIONS,
                          "seeds": SEEDS, "epochs": EPOCHS,
                          "train_n": len(clean_labels), "test_n": len(bundle.test_y)},
               "timestamp": run_stamp(), "runs": [], "detection": []}

    for frac in FRACTIONS:
        for seed in SEEDS:
            run_rng = np.random.default_rng(1000 + seed)
            poisoned_y, flip_idx = poison_label_flips(
                bundle.train_raw_x, bundle.train_raw_y, SOURCE, TARGET, frac, run_rng)
            train_x = to_float(bundle.train_raw_x)
            model = make_model("cnn")
            hist = train_model(model, train_x, poisoned_y, epochs=EPOCHS, seed=seed,
                               log_prefix=f"[poison f={frac} s={seed}] ",
                               test_x=bundle.test_x, test_y=bundle.test_y)
            preds, probs, _ = evaluate(model, bundle.test_x, bundle.test_y)
            acc = accuracy(preds, bundle.test_y)
            pc = {}
            for c in range(10):
                m = bundle.test_y == c
                pc[c] = float((preds[m] == c).mean())
            ece = expected_calibration_error(probs, bundle.test_y)
            src_acc = pc[SOURCE]
            # confusion mass transferred from SOURCE to TARGET (off-diagonal row SOURCE)
            row_src = preds[bundle.test_y == SOURCE]
            transfer = float((row_src == TARGET).mean())
            results["runs"].append({
                "fraction": frac, "seed": seed,
                "clean_accuracy": acc, "accuracy_ci95": list(binomial_ci(acc, len(preds))),
                "source_class_accuracy": src_acc, "target_class_accuracy": pc[TARGET],
                "src_to_tgt_transfer_rate": transfer,
                "ece": ece, "train_seconds": hist["train_seconds"],
                "final_train_loss": hist["train_loss"][-1],
                "per_class": {str(k): v for k, v in pc.items()},
                "confusion": None,  # keep JSON small; last seed stores it below
            })
            if seed == 2:
                from amlsec.metrics import confusion_matrix
                results["runs"][-1]["confusion"] = confusion_matrix(preds, bundle.test_y).tolist()
            print(f"[poison] f={frac} seed={seed} acc={acc:.4f} src_acc={src_acc:.4f} "
                  f"ece={ece:.4f} transfer={transfer:.3f}", flush=True)

    # ---- detection-before-degradation: audit the poisoned datasets at each fraction
    for frac in FRACTIONS:
        if frac == 0.0:
            continue
        run_rng = np.random.default_rng(1000)
        poisoned_y, flip_idx = poison_label_flips(
            bundle.train_raw_x, bundle.train_raw_y, SOURCE, TARGET, frac, run_rng)
        audit_scores = knn_label_audit(bundle.train_raw_x, poisoned_y, k=5)
        flip_set = set(flip_idx.tolist())
        n_flag = max(1, int(0.01 * len(poisoned_y)))
        flagged = set(np.argsort(-audit_scores)[:n_flag])
        recall = len(flagged & flip_set) / max(1, len(flip_idx))
        # clean-only baseline separation
        sep = float(np.mean(audit_scores[flip_idx]) - np.mean(np.delete(audit_scores, flip_idx)))
        results["detection"].append({
            "fraction": frac, "recall_at_1pct_flagged": recall,
            "mean_disagreement_poisoned": float(np.mean(audit_scores[flip_idx])),
            "mean_disagreement_benign": float(np.mean(np.delete(audit_scores, flip_idx))),
            "separation": sep,
            "clean_baseline_mean": float(np.mean(clean_audit)),
        })
        print(f"[poison-detect] f={frac} recall@1%={recall:.3f} separation={sep:.4f}", flush=True)

    # ---- defense: audit-and-filter before retraining (at the worst fraction, 3 seeds)
    defense_runs = []
    for frac in (0.05, 0.10):
        for seed in SEEDS:
            run_rng = np.random.default_rng(1000 + seed)
            poisoned_y, flip_idx = poison_label_flips(
                bundle.train_raw_x, bundle.train_raw_y, SOURCE, TARGET, frac, run_rng)
            fx, fy, n_drop, _ = audit_and_filter(bundle.train_raw_x, poisoned_y,
                                                 keep_frac=1 - min(0.05, frac * 1.2))
            model = make_model("cnn")
            train_model(model, to_float(fx), fy, epochs=EPOCHS, seed=seed,
                        log_prefix=f"[poison-def f={frac} s={seed}] ")
            preds, probs, _ = evaluate(model, bundle.test_x, bundle.test_y)
            acc = accuracy(preds, bundle.test_y)
            src_acc = float((preds[bundle.test_y == SOURCE] == SOURCE).mean())
            defense_runs.append({"fraction": frac, "seed": seed, "accuracy": acc,
                                 "source_class_accuracy": src_acc,
                                 "dropped": n_drop})
            print(f"[poison-def] f={frac} s={seed} acc={acc:.4f} src_acc={src_acc:.4f}", flush=True)
    results["defense_filter"] = defense_runs
    save_result("poisoning", results)


if __name__ == "__main__":
    main()