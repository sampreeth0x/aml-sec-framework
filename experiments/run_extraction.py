"""Experiment C — model extraction under different API policies.

Attacker query pool: the 4,000-image test population (disjoint from target's
training data — a realistic attacker owns same-distribution unlabeled data).
Substitute models are trained on API responses only.

Sweep: query budget {500, 1000, 2000, 4000} x policy {full_confidence, rounded,
argmax_only}. Metrics: substitute accuracy on a held-out slice, fidelity
(agreement with target), queries used.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from amlsec.attacks.extraction import PredictionAPI, RateLimitedAPI, extraction_attack
from amlsec.experiment_utils import save_result, run_stamp, new_bundle
from amlsec.metrics import accuracy
from amlsec.models import make_model, train_model, evaluate

BUDGETS = [500, 1000, 2000, 4000]
POLICIES = ["full_confidence", "rounded", "argmax_only"]
SEEDS = [0, 1, 2]
EPOCHS = 6


def main():
    bundle = new_bundle()
    results = {"config": {"budgets": BUDGETS, "policies": POLICIES, "sub_epochs": EPOCHS},
               "timestamp": run_stamp(), "target": {}, "runs": []}

    # ---- train the real target model once (seed 0)
    target = make_model("cnn")
    hist = train_model(target, bundle.train_x, bundle.train_y, epochs=4, seed=0,
                       log_prefix="[target] ", test_x=bundle.test_x, test_y=bundle.test_y)
    tgt_preds, _, _ = evaluate(target, bundle.test_x, bundle.test_y)
    results["target"] = {"accuracy": accuracy(tgt_preds, bundle.test_y),
                         "train_seconds": hist["train_seconds"]}

    # attacker's unlabeled pool = test images; fidelity measured on a disjoint half
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(bundle.test_x))
    query_pool_x, query_pool_y = bundle.test_x[perm[:3000]], bundle.test_y[perm[:3000]]
    eval_x, eval_y = bundle.test_x[perm[3000:]], bundle.test_y[perm[3000:]]
    tgt_eval_preds, _, _ = evaluate(target, eval_x, eval_y)
    target_acc_eval = accuracy(tgt_eval_preds, eval_y)

    for budget in BUDGETS:
        for policy in POLICIES:
            accs, fids = [], []
            for seed in SEEDS:
                api = PredictionAPI(target, policy=policy)
                sub, responses, sub_hist, qtime = extraction_attack(
                    api, query_pool_x, query_pool_y, lambda: make_model("cnn"),
                    substitute_epochs=EPOCHS, seed=seed, max_queries=budget)
                sub_preds, _, _ = evaluate(sub, eval_x, eval_y)
                sub_acc = accuracy(sub_preds, eval_y)
                fidelity = float((sub_preds == tgt_eval_preds).mean())
                accs.append(sub_acc)
                fids.append(fidelity)
                print(f"[extract] budget={budget} policy={policy} s={seed} "
                      f"sub_acc={sub_acc:.4f} fidelity={fidelity:.4f}", flush=True)
            results["runs"].append({
                "budget": budget, "policy": policy,
                "queries_used": min(budget, len(query_pool_x)),
                "substitute_accuracy_mean": float(np.mean(accs)),
                "substitute_accuracy_std": float(np.std(accs)),
                "fidelity_mean": float(np.mean(fids)),
                "fidelity_std": float(np.std(fids)),
            })

    # ---- rate-limited API: attacker simply cannot finish extraction
    for budget in BUDGETS:
        api = RateLimitedAPI(target, "full_confidence", max_queries=1000)
        sub, responses, _, _ = extraction_attack(
            api, query_pool_x, query_pool_y, lambda: make_model("cnn"),
            substitute_epochs=EPOCHS, seed=0, max_queries=budget)
        sub_preds, _, _ = evaluate(sub, eval_x, eval_y)
        results["runs"].append({
            "budget": budget, "policy": "full_confidence+ratelimit1000",
            "queries_used": api.query_count,
            "refused_queries": api.refused_queries,
            "substitute_accuracy_mean": accuracy(sub_preds, eval_y),
            "substitute_accuracy_std": 0.0,
            "fidelity_mean": float((sub_preds == tgt_eval_preds).mean()),
            "fidelity_std": 0.0,
        })
        print(f"[extract] budget={budget} ratelimit: used={api.query_count} "
              f"refused={api.refused_queries}", flush=True)

    results["target_accuracy_on_eval_slice"] = target_acc_eval
    save_result("extraction", results)


if __name__ == "__main__":
    main()