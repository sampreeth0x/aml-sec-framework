"""Experiment E — adversarial examples: eps sweep + adversarial training defense.

Baseline CNN vs PGD-adversarially-trained CNN. Both attacked with FGSM and
PGD over an eps grid. Produces the robustness-accuracy trade-off data for the
Pareto analysis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from amlsec.attacks.adversarial import fgsm, pgd
from amlsec.data import to_float
from amlsec.defenses import pgd_adversarial_training
from amlsec.experiment_utils import save_result, run_stamp, new_bundle
from amlsec.metrics import accuracy
from amlsec.models import make_model, train_model, evaluate

EPS_GRID = [0.02, 0.05, 0.1, 0.2]
SEEDS = [0, 1, 2]


def eval_model(model, bundle):
    """Clean accuracy + FGSM/PGD robust accuracy across the eps grid."""
    preds, probs, _ = evaluate(model, bundle.test_x, bundle.test_y)
    clean = accuracy(preds, bundle.test_y)
    rows = {"clean_accuracy": clean, "attacks": []}
    test_raw = bundle.test_raw_x.astype(np.float32)[:, None, :, :] / 255.0  # NCHW
    for eps in EPS_GRID:
        for name, attack in (("fgsm", fgsm), ("pgd", pgd)):
            accs = []
            for seed in SEEDS:
                x_adv = attack(model, test_raw, bundle.test_y, eps, seed=seed)
                p_adv, _, _ = evaluate(model, (x_adv - 0.1307) / 0.3081, bundle.test_y)
                accs.append(accuracy(p_adv, bundle.test_y))
            rows["attacks"].append({
                "attack": name, "eps": eps,
                "robust_accuracy_mean": float(np.mean(accs)),
                "robust_accuracy_std": float(np.std(accs)),
                "attack_success_rate": float(1 - np.mean(accs)),
            })
            print(f"[adv] {name} eps={eps} robust={np.mean(accs):.4f}", flush=True)
    return rows


def main():
    bundle = new_bundle()
    results = {"timestamp": run_stamp(), "config": {"eps_grid": EPS_GRID}, "models": {}}

    # ---------- baseline models
    baseline_rows = []
    for seed in SEEDS:
        model = make_model("cnn")
        train_model(model, bundle.train_x, bundle.train_y, epochs=4, seed=seed,
                    log_prefix=f"[adv-base s={seed}] ", test_x=bundle.test_x, test_y=bundle.test_y)
        r = eval_model(model, bundle)
        r["seed"] = seed
        baseline_rows.append(r)
    results["models"]["baseline"] = baseline_rows

    # ---------- adversarially trained models
    at_rows = []
    for seed in SEEDS:
        model = make_model("cnn")
        pgd_adversarial_training(model, bundle.train_raw_x, bundle.train_raw_y,
                                 epochs=3, eps=0.1, steps=4, seed=seed,
                                 log_prefix=f"[adv-AT s={seed}] ")
        r = eval_model(model, bundle)
        r["seed"] = seed
        at_rows.append(r)
    results["models"]["pgd_adversarial_training"] = at_rows

    # ---------- regularization-only model (label smoothing 0.1 as a cheap defense)
    from amlsec.defenses import label_smoothing_train
    ls_rows = []
    for seed in SEEDS:
        model = make_model("cnn")
        label_smoothing_train(model, bundle.train_x, bundle.train_y, epochs=4,
                              smoothing=0.1, seed=seed)
        r = eval_model(model, bundle)
        r["seed"] = seed
        ls_rows.append(r)
    results["models"]["label_smoothing"] = ls_rows

    save_result("adversarial", results)


if __name__ == "__main__":
    main()