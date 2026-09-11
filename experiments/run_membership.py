"""Experiment D — membership inference + overfitting/leakage correlation.

Study 1 (overfitting): same CNN trained for epochs {1, 2, 4, 8}; the MIA
confidence-threshold attack runs on members (train set) vs non-members
(test population). Correlates the train/test accuracy gap with leakage AUC.

Study 2 (defenses): label smoothing, DP-style clipped+noisy training, and
early stopping; each reports clean accuracy alongside attack AUC — the
privacy-utility trade-off, measured.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from amlsec.attacks.membership import run_membership_attack, attack_scores
from amlsec.defenses import label_smoothing_train, make_dp_hook
from amlsec.experiment_utils import save_result, run_stamp, new_bundle
from amlsec.metrics import accuracy, roc_auc
from amlsec.models import make_model, train_model, evaluate

EPOCH_GRID = [1, 2, 4, 8]
SEEDS = [0, 1, 2]


def mia_for_model(model, bundle, member_frac=0.5, seed=0):
    """Members: half of the real training set; non-members: test population.
    Both come from the same generative distribution (standard MIA setup)."""
    rng = np.random.default_rng(42 + seed)
    n = len(bundle.train_x)
    member_idx = rng.permutation(n)[: int(member_frac * n)]
    mx, my = bundle.train_x[member_idx], bundle.train_y[member_idx]
    return run_membership_attack(model, mx, my, bundle.test_x, bundle.test_y)


def main():
    bundle = new_bundle()
    results = {"timestamp": run_stamp(), "overfitting_study": [], "defenses": [],
               "config": {"epoch_grid": EPOCH_GRID, "seeds": SEEDS}}

    # ---------- study 1: leakage vs overfitting
    for ep in EPOCH_GRID:
        per_seed = []
        for seed in SEEDS:
            model = make_model("cnn")
            hist = train_model(model, bundle.train_x, bundle.train_y, epochs=ep, seed=seed,
                               log_prefix=f"[mia ep={ep} s={seed}] ")
            preds, _, _ = evaluate(model, bundle.train_x, bundle.train_y)
            tr_acc = accuracy(preds, bundle.train_y)
            preds_t, _, _ = evaluate(model, bundle.test_x, bundle.test_y)
            te_acc = accuracy(preds_t, bundle.test_y)
            mia = mia_for_model(model, bundle, seed=seed)
            per_seed.append({
                "seed": seed, "epochs": ep,
                "train_accuracy": tr_acc, "test_accuracy": te_acc,
                "overfit_gap": tr_acc - te_acc,
                "mia": mia,
            })
            print(f"[mia] ep={ep} s={seed} gap={tr_acc - te_acc:.4f} "
                  f"best_auc={mia[mia['best']]['auc']:.4f}", flush=True)
        results["overfitting_study"].append(per_seed)

    # ---------- study 2: defenses
    # (a) early stopping = fewer epochs (already covered by ep=1,2; reference)
    # (b) label smoothing
    for smoothing in (0.1, 0.3):
        accs, aucs = [], []
        for seed in SEEDS:
            model = make_model("cnn")
            label_smoothing_train(model, bundle.train_x, bundle.train_y, epochs=4,
                                  smoothing=smoothing, seed=seed)
            preds, _, _ = evaluate(model, bundle.test_x, bundle.test_y)
            accs.append(accuracy(preds, bundle.test_y))
            mia = mia_for_model(model, bundle, seed=seed)
            aucs.append(mia[mia["best"]]["auc"])
        results["defenses"].append({
            "defense": "label_smoothing", "param": smoothing,
            "clean_accuracy_mean": float(np.mean(accs)),
            "mia_auc_mean": float(np.mean(aucs)),
            "baseline_auc_mean": float(np.mean([s["mia"][s["mia"]["best"]]["auc"]
                                                for grp in results["overfitting_study"]
                                                if grp[0]["epochs"] == 4 for s in grp])),
        })
        print(f"[mia-def] LS={smoothing} acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f}", flush=True)

    # (c) DP-style training (clipped + noisy gradients)
    for noise in (0.5, 1.0, 2.0):
        accs, aucs = [], []
        for seed in SEEDS:
            model = make_model("cnn")
            hook = make_dp_hook(grad_clip=1.0, noise_multiplier=noise, batch_size=64, seed=seed)
            train_model(model, bundle.train_x, bundle.train_y, epochs=4, seed=seed,
                        per_batch_hook=hook, log_prefix=f"[dp n={noise} s={seed}] ",
                        test_x=bundle.test_x, test_y=bundle.test_y)
            preds, _, _ = evaluate(model, bundle.test_x, bundle.test_y)
            accs.append(accuracy(preds, bundle.test_y))
            mia = mia_for_model(model, bundle, seed=seed)
            aucs.append(mia[mia["best"]]["auc"])
        results["defenses"].append({
            "defense": "dp_grad_clip+noise", "param": noise,
            "clean_accuracy_mean": float(np.mean(accs)),
            "mia_auc_mean": float(np.mean(aucs)),
            "baseline_auc_mean": 0.5,  # filled in final analysis
        })
        print(f"[mia-def] DP noise={noise} acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f}", flush=True)

    save_result("membership", results)


if __name__ == "__main__":
    main()