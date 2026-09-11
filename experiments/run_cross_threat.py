"""Cross-threat security matrix — does defending one threat change exposure to others?

Design: a defense is a TRAINING REGIME. Each regime is trained three ways
(clean, 5% label-flip poisoned, 5% backdoored) x 3 seeds. Every clean-trained
regime model is measured against extraction / MIA / PGD; every poison-trained
regime against poisoning metrics; every backdoor-trained regime against ASR.

Raw measured metrics are primary; the composite S and delta-S are computed
afterwards and kept secondary (documented formula, fixed weights).

Regimes:
  baseline     plain 4-epoch training
  auditfilter  kNN audit-and-filter (keep 98%) applied to whatever data it gets
  at           PGD adversarial training (3 ep, eps=0.1, 4 steps)
  dp           DP-style: batch grad clip 1.0 + Gaussian noise multiplier 1.0
  ls           label smoothing 0.1
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from amlsec.attacks.backdoor import build_backdoor_dataset, build_triggered_test
from amlsec.attacks.extraction import PredictionAPI, train_soft
from amlsec.attacks.poisoning import knn_label_audit, poison_label_flips
from amlsec.data import to_float
from amlsec.defenses import audit_and_filter, make_dp_hook, pgd_adversarial_training
from amlsec.experiment_utils import save_result, run_stamp, new_bundle
from amlsec.metrics import accuracy, expected_calibration_error, roc_auc
from amlsec.models import make_model, train_model, evaluate

SEEDS = [0, 1, 2]
POISON_FRAC = 0.05
BD_FRAC = 0.05
BD_TARGET = 0
SOURCE, TARGET = 2, 7
EPS_EVAL = 0.1
BUDGET = 4000


def train_regime(regime, raw_x, labels, seed):
    """Train one (regime, data) pair. Returns (model, seconds)."""
    t0 = time.time()
    if regime == "at":
        model = make_model("cnn")
        pgd_adversarial_training(model, raw_x, labels, epochs=3, eps=EPS_EVAL, steps=4,
                                 seed=seed, log_prefix=f"[X at s={seed}] ")
    elif regime == "auditfilter":
        fx, fy, _, _ = audit_and_filter(raw_x, labels, keep_frac=0.98, k=5)
        model = make_model("cnn")
        train_model(model, to_float(fx), fy, epochs=4, seed=seed,
                    log_prefix=f"[X filt s={seed}] ")
    elif regime == "dp":
        model = make_model("cnn")
        hook = make_dp_hook(grad_clip=1.0, noise_multiplier=1.0, batch_size=64, seed=seed)
        train_model(model, to_float(raw_x), labels, epochs=4, seed=seed,
                    per_batch_hook=hook, log_prefix=f"[X dp s={seed}] ")
    elif regime == "ls":
        model = make_model("cnn")
        from amlsec.defenses import label_smoothing_train
        label_smoothing_train(model, to_float(raw_x), labels, epochs=4, smoothing=0.1, seed=seed)
    else:
        model = make_model("cnn")
        train_model(model, to_float(raw_x), labels, epochs=4, seed=seed,
                    log_prefix=f"[X base s={seed}] ")
    return model, time.time() - t0


def mia_auc(model, bundle, seed):
    rng = np.random.default_rng(42 + seed)
    n = len(bundle.train_x)
    m_idx = rng.permutation(n)[: n // 2]
    from amlsec.attacks.membership import run_membership_attack
    mia = run_membership_attack(model, bundle.train_x[m_idx], bundle.train_y[m_idx],
                                bundle.test_x, bundle.test_y)
    return mia[mia["best"]]["auc"]


def extraction_fidelity(model, bundle, seed):
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(bundle.test_x))
    pool_x, eval_x, eval_y = bundle.test_x[perm[:3000]], bundle.test_x[perm[3000:]], bundle.test_y[perm[3000:]]
    api = PredictionAPI(model, "full_confidence")
    tgt_preds, _, _ = evaluate(model, eval_x, eval_y)
    responses = []
    for i in range(0, BUDGET, 500):
        responses.append(api.query(pool_x[i:i + 500]))
    responses = np.vstack(responses)
    sub = make_model("cnn")
    train_soft(sub, pool_x, __import__("torch").from_numpy(responses), epochs=6, seed=seed)
    sub_preds, _, _ = evaluate(sub, eval_x, eval_y)
    return float((sub_preds == tgt_preds).mean()), api.query_count


def pgd_success(model, bundle, seed):
    from amlsec.attacks.adversarial import pgd
    test_raw = bundle.test_raw_x.astype(np.float32)[:, None, :, :] / 255.0
    x_adv = pgd(model, test_raw, bundle.test_y, eps=EPS_EVAL, seed=seed)
    p_adv, _, _ = evaluate(model, (x_adv - 0.1307) / 0.3081, bundle.test_y)
    return 1 - accuracy(p_adv, bundle.test_y)


def main():
    bundle = new_bundle()
    clean_y = bundle.train_raw_y.copy()
    regimes = ["baseline", "auditfilter", "at", "dp", "ls"]

    results = {"timestamp": run_stamp(), "config": {
        "regimes": regimes, "seeds": SEEDS, "poison_frac": POISON_FRAC,
        "bd_frac": BD_FRAC, "bd_target": BD_TARGET, "eps_eval": EPS_EVAL, "budget": BUDGET},
        "scenarios": {}}

    rng = np.random.default_rng(1000)
    pois_y, _ = poison_label_flips(bundle.train_raw_x, clean_y, SOURCE, TARGET,
                                   POISON_FRAC, rng)
    trig_test_x = to_float(build_triggered_test(bundle, BD_TARGET))
    bd_x, bd_y, bd_idx = build_backdoor_dataset(bundle, poison_frac=BD_FRAC,
                                                target_class=BD_TARGET, seed=0)

    # scenario datasets (raw pixel space)
    scenarios = {
        "clean": (bundle.train_raw_x, clean_y),
        "poison5": (bundle.train_raw_x, pois_y),
        "backdoor5": (bd_x, bd_y),
    }

    for regime in regimes:
        rows = {"regime": regime, "clean": [], "poison": [], "backdoor": [],
                "mia": [], "extraction": [], "adv": []}
        for seed in SEEDS:
            # ---- clean-trained model: measure cross-threat exposure
            model, secs = train_regime(regime, *scenarios["clean"], seed)
            preds, probs, _ = evaluate(model, bundle.test_x, bundle.test_y)
            row = {"seed": seed, "train_seconds": secs,
                   "clean_accuracy": accuracy(preds, bundle.test_y),
                   "ece": expected_calibration_error(probs, bundle.test_y),
                   "mia_auc": mia_auc(model, bundle, seed),
                   "extraction_fidelity": extraction_fidelity(model, bundle, seed)[0],
                   "pgd_success": pgd_success(model, bundle, seed)}
            rows["clean"].append(row)
            print(f"[X] {regime} s={seed} clean_acc={row['clean_accuracy']:.4f} "
                  f"mia={row['mia_auc']:.4f} fid={row['extraction_fidelity']:.4f} "
                  f"pgd_succ={row['pgd_success']:.4f}", flush=True)

            # ---- poison-trained model of the same regime
            model_p, secs_p = train_regime(regime, *scenarios["poison5"], seed)
            preds_p, probs_p, _ = evaluate(model_p, bundle.test_x, bundle.test_y)
            src_mask = bundle.test_y == SOURCE
            rows["poison"].append({
                "seed": seed, "train_seconds": secs_p,
                "clean_accuracy": accuracy(preds_p, bundle.test_y),
                "src_class_accuracy": float((preds_p[src_mask] == SOURCE).mean()),
                "src_to_tgt_transfer": float((preds_p[src_mask] == TARGET).mean()),
                "ece": expected_calibration_error(probs_p, bundle.test_y)})

            # ---- backdoor-trained model of the same regime
            model_b, secs_b = train_regime(regime, *scenarios["backdoor5"], seed)
            trig_preds, _, _ = evaluate(model_b, trig_test_x, bundle.test_y)
            preds_b, _, _ = evaluate(model_b, bundle.test_x, bundle.test_y)
            rows["backdoor"].append({
                "seed": seed, "train_seconds": secs_b,
                "clean_accuracy": accuracy(preds_b, bundle.test_y),
                "asr": float((trig_preds == BD_TARGET).mean())})

        save_regime(rows, regime)
        print(f"[X] {regime} DONE", flush=True)


def save_regime(rows, regime):
    save_result(f"cross_{regime}", rows)


if __name__ == "__main__":
    main()