"""Experiment B — backdoor insertion + stealth analysis + detection.

For poison fractions {1, 3, 5}% (3 seeds): train on trigger-poisoned data,
measure clean accuracy vs attack success rate, then run two real detectors:
  1. trigger-flip scan on the deployed model
  2. data audit: correlation between trigger-region presence and label anomalies
Defense: audit-and-filter retraining (drop suspicious samples, retrain clean).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from amlsec.attacks.backdoor import build_backdoor_dataset, build_triggered_test, stamp_trigger
from amlsec.data import to_float
from amlsec.defenses import audit_and_filter
from amlsec.experiment_utils import save_result, run_stamp, new_bundle
from amlsec.metrics import accuracy
from amlsec.models import make_model, train_model, evaluate

TARGET_CLASS = 0
BOX = (0, 0, 3, 3)  # 3x3 solid patch top-left
FRACTIONS = [0.0, 0.01, 0.03, 0.05]
SEEDS = [0, 1, 2]
EPOCHS = 4


def main():
    bundle = new_bundle()
    results = {"config": {"target_class": TARGET_CLASS, "box": list(BOX),
                          "fractions": FRACTIONS, "seeds": SEEDS},
               "timestamp": run_stamp(), "runs": [], "defense": []}

    trig_test_x = to_float(build_triggered_test(bundle, TARGET_CLASS, BOX))

    for frac in FRACTIONS:
        for seed in SEEDS:
            px, py, poison_idx = build_backdoor_dataset(
                bundle, poison_frac=frac, target_class=TARGET_CLASS, box=BOX, seed=seed)
            model = make_model("cnn")
            hist = train_model(model, to_float(px), py, epochs=EPOCHS, seed=seed,
                               log_prefix=f"[bd f={frac} s={seed}] ",
                               test_x=bundle.test_x, test_y=bundle.test_y)
            preds, _, _ = evaluate(model, bundle.test_x, bundle.test_y)
            clean_acc = accuracy(preds, bundle.test_y)
            trig_preds, _, _ = evaluate(model, trig_test_x, bundle.test_y)
            asr = float((trig_preds == TARGET_CLASS).mean())
            # flip-scan detection signal
            from amlsec.attacks.backdoor import trigger_flip_scan
            scan = trigger_flip_scan(model, bundle.test_raw_x, bundle.test_y,
                                     evaluate, TARGET_CLASS, BOX)
            results["runs"].append({
                "fraction": frac, "seed": seed, "clean_accuracy": clean_acc,
                "attack_success_rate": asr, "stealth_gap": None,
                "trigger_flip_scan": scan, "train_seconds": hist["train_seconds"],
            })
            print(f"[bd] f={frac} s={seed} clean={clean_acc:.4f} ASR={asr:.4f} "
                  f"flip_scan={scan['flip_rate_to_target']:.3f}", flush=True)

    # stealth gap: clean acc relative to the f=0 baseline mean
    base = np.mean([r["clean_accuracy"] for r in results["runs"] if r["fraction"] == 0.0])
    for r in results["runs"]:
        r["stealth_gap"] = r["clean_accuracy"] - base

    # ---- defense: audit-and-filter then retrain (5% backdoor, 3 seeds)
    for seed in SEEDS:
        px, py, poison_idx = build_backdoor_dataset(
            bundle, poison_frac=0.05, target_class=TARGET_CLASS, box=BOX, seed=seed)
        fx, fy, n_drop, dropped = audit_and_filter(px, py, keep_frac=0.95, k=5)
        recovered = len(set(dropped.tolist()) & set(poison_idx.tolist())) / len(poison_idx)
        model = make_model("cnn")
        train_model(model, to_float(fx), fy, epochs=EPOCHS, seed=seed,
                    log_prefix=f"[bd-def s={seed}] ")
        preds, _, _ = evaluate(model, bundle.test_x, bundle.test_y)
        trig_preds, _, _ = evaluate(model, trig_test_x, bundle.test_y)
        results["defense"].append({
            "seed": seed, "dropped": n_drop, "poison_recovered": recovered,
            "clean_accuracy": accuracy(preds, bundle.test_y),
            "attack_success_rate": float((trig_preds == TARGET_CLASS).mean()),
        })
        print(f"[bd-def] s={seed} dropped={n_drop} recovered={recovered:.3f} "
              f"clean={results['defense'][-1]['clean_accuracy']:.4f} "
              f"ASR={results['defense'][-1]['attack_success_rate']:.4f}", flush=True)

    save_result("backdoor", results)


if __name__ == "__main__":
    main()