"""Membership inference on a genuinely-overfit regime.

The 20k-sample study showed AUC ~0.50 — models trained on plenty of data don't
memorize enough to leak. This arm repeats the overfitting study on a 3k-sample
subset (300/class), where the train/test gap is large and membership leakage
should become measurable. Also runs the DP-style defense there, where the
privacy-utility trade-off is actually testable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from amlsec.attacks.membership import run_membership_attack
from amlsec.defenses import label_smoothing_train, make_dp_hook
from amlsec.experiment_utils import save_result, run_stamp, new_bundle
from amlsec.metrics import accuracy
from amlsec.models import make_model, train_model, evaluate

SEEDS = [0, 1, 2]


def mia_for_model(model, bundle, member_x, member_y, seed=0):
    return run_membership_attack(model, member_x, member_y, bundle.test_x, bundle.test_y)


def main():
    bundle = new_bundle()
    rng = np.random.default_rng(42)
    n = len(bundle.train_x)
    member_idx = rng.permutation(n)[:1500]      # members: 1500 of the trained samples
    mx, my = bundle.train_x[member_idx], bundle.train_y[member_idx]

    results = {"timestamp": run_stamp(), "runs": []}

    configs = [("baseline", dict(epochs=8, batch_size=32)),
               ("baseline_long", dict(epochs=20, batch_size=32)),
               ("dp_noise1.0", dict(epochs=20, batch_size=32, dp_noise=1.0)),
               ("dp_noise3.0", dict(epochs=20, batch_size=32, dp_noise=3.0))]
    for name, cfg in configs:
        for seed in SEEDS:
            model = make_model("cnn")
            hook = None
            if "dp_noise" in cfg:
                hook = make_dp_hook(grad_clip=1.0, noise_multiplier=cfg["dp_noise"],
                                    batch_size=cfg["batch_size"], seed=seed)
            train_model(model, bundle.train_x, bundle.train_y,
                        epochs=cfg["epochs"], batch_size=cfg["batch_size"], seed=seed,
                        per_batch_hook=hook, log_prefix=f"[miaS {name} s={seed}] ",
                        test_x=bundle.test_x, test_y=bundle.test_y)
            preds, _, _ = evaluate(model, bundle.train_x, bundle.train_y)
            tr_acc = accuracy(preds, bundle.train_y)
            preds_t, _, _ = evaluate(model, bundle.test_x, bundle.test_y)
            te_acc = accuracy(preds_t, bundle.test_y)
            mia = mia_for_model(model, bundle, mx, my, seed=seed)
            results["runs"].append({
                "config": name, "seed": seed, "train_accuracy": tr_acc,
                "test_accuracy": te_acc, "overfit_gap": tr_acc - te_acc, "mia": mia,
            })
            print(f"[miaS] {name} s={seed} gap={tr_acc - te_acc:.4f} "
                  f"auc={mia[mia['best']]['auc']:.4f}", flush=True)

    save_result("membership_small", results)


if __name__ == "__main__":
    main()