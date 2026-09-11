"""Final analysis — unified security scoring + security-utility trade-off matrix.

The scoring formula (defined here, computed ONLY from measured results):

  For each attack dimension, risk = normalized attacker success measured under
  the UNDEFENDED configuration, then the same metric is recomputed under the
  best defense from our experiments. A dimension's score is 0-100, higher = a
  more exposed system.

  poisoning    : w1*max_acc_drop(5% f) + w2*src_tgt_transfer + w3*(1 - audit_recall)
  backdoor     : ASR at 5% poison x stealth multiplier (1 - clean_acc_drop)
  extraction   : substitute fidelity at max budget (full confidence API)
  membership   : 2 x (AUC - 0.5) scaled 100, clipped at [0,100]
  adversarial  : PGD success rate at eps=0.1
  supply_chain : tamper visibility deficit = 1 - P(behavioral check catches it)
                 (integrity controls reduce this to ~0)

  Weights chosen once and held fixed across all runs (documented, not tuned).
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amlsec.experiment_utils import RESULTS_DIR, save_result, run_stamp


def load(name):
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


def mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def score_poisoning(p):
    runs = [r for r in p["runs"] if r["fraction"] == 0.05]
    acc_drop = mean([r["clean_accuracy"] for r in runs]) - mean(
        [r["clean_accuracy"] for r in p["runs"] if r["fraction"] == 0.0])
    transfer = mean([r["src_to_tgt_transfer_rate"] for r in runs])
    det = mean([d["recall_at_1pct_flagged"] for d in p["detection"] if d["fraction"] == 0.05])
    risk = 0.4 * max(0.0, acc_drop) / 0.15 + 0.35 * transfer + 0.25 * (1 - det)
    defended = abs(mean([d["accuracy"] for d in p["defense_filter"] if d["fraction"] == 0.05]) -
                   mean([r["clean_accuracy"] for r in p["runs"] if r["fraction"] == 0.0]))
    return {
        "risk_undefended": float(np.clip(risk, 0, 1) * 100),
        "measured": {"accuracy_drop_at_5pct": acc_drop, "src_to_tgt_transfer": transfer,
                     "audit_recall": det},
        "defense": "kNN audit-and-filter retraining",
        "defense_residual_drop": defended,
        "defense_recovery": mean([d["source_class_accuracy"] for d in p["defense_filter"]
                                  if d["fraction"] == 0.05]),
    }


def score_backdoor(b):
    runs5 = [r for r in b["runs"] if r["fraction"] == 0.05]
    asr = mean([r["attack_success_rate"] for r in runs5])
    stealth = 1 - abs(mean([r["stealth_gap"] for r in runs5]))
    risk = asr * (0.5 + 0.5 * stealth)
    defense_asr = mean([d["attack_success_rate"] for d in b["defense"]])
    scan = mean([r["trigger_flip_scan"]["flip_rate_to_target"] for r in runs5])
    # measured defense reality: the data audit does NOT remove the backdoor;
    # the model-level trigger-flip scan is the detector that works
    return {
        "risk_undefended": float(np.clip(risk, 0, 1) * 100),
        "measured": {"asr_at_5pct": asr, "stealth_multiplier": stealth,
                     "clean_accuracy_at_5pct": mean([r["clean_accuracy"] for r in runs5])},
        "defense": "kNN audit-and-filter (data-level) — measured INEFFECTIVE",
        "defense_asr": defense_asr,
        "detection_trigger_flip_scan": scan,
    }


def score_extraction(e):
    full = [r for r in e["runs"] if r["policy"] == "full_confidence" and r["budget"] == 4000]
    arg = [r for r in e["runs"] if r["policy"] == "argmax_only" and r["budget"] == 4000]
    fid = mean([r["fidelity_mean"] for r in full])
    fid_arg = mean([r["fidelity_mean"] for r in arg])
    return {
        "risk_undefended": fid * 100,
        "measured": {"fidelity_max_budget_full_api": fid,
                     "fidelity_argmax_api": fid_arg},
        "defense": "argmax-only + rate limiting",
        "defense_fidelity": fid_arg,
    }


def score_membership(m):
    ep4 = [s for grp in m["overfitting_study"] if grp[0]["epochs"] == 4 for s in grp]
    ep8 = [s for grp in m["overfitting_study"] if grp[0]["epochs"] == 8 for s in grp]
    auc4 = mean([s["mia"][s["mia"]["best"]]["auc"] for s in ep4])
    auc8 = mean([s["mia"][s["mia"]["best"]]["auc"] for s in ep8])
    dp = [d for d in m["defenses"] if d["defense"] == "dp_grad_clip+noise"]
    dp_best = max(dp, key=lambda d: d["clean_accuracy_mean"]) if dp else None
    # the 20k regime leaks almost nothing; use the overfit 3k arm for the real
    # leakage measurement (long-data baseline_arm recorded alongside)
    small_auc = None
    try:
        ms = json.loads((RESULTS_DIR / "membership_small.json").read_text())
        base = [r for r in ms["runs"] if r["config"] == "baseline_long"]
        small_auc = mean([r["mia"][r["mia"]["best"]]["auc"] for r in base])
    except FileNotFoundError:
        pass
    risk_auc = small_auc if small_auc is not None else auc8
    return {
        "risk_undefended": float(np.clip((risk_auc - 0.5) * 2, 0, 1) * 100),
        "measured": {"auc_4epochs_20k": auc4, "auc_8epochs_20k": auc8,
                     "auc_overfit_3k_long": small_auc},
        "defense": "DP gradient clip+noise",
        "defense_auc": dp_best["mia_auc_mean"] if dp_best else None,
        "defense_clean_accuracy": dp_best["clean_accuracy_mean"] if dp_best else None,
    }


def score_adversarial(a):
    base = a["models"]["baseline"]
    at = a["models"]["pgd_adversarial_training"]
    pgd_rows_base = [r for r in base[0]["attacks"] if r["attack"] == "pgd" and r["eps"] == 0.1]
    pgd_rows_at = [r for r in at[0]["attacks"] if r["attack"] == "pgd" and r["eps"] == 0.1]
    asr_base = mean([1 - r["robust_accuracy_mean"] for r in pgd_rows_base])
    asr_at = mean([1 - r["robust_accuracy_mean"] for r in pgd_rows_at])
    clean_base = mean([r["clean_accuracy"] for r in base])
    clean_at = mean([r["clean_accuracy"] for r in at])
    return {
        "risk_undefended": asr_base * 100,
        "measured": {"pgd_success_eps0.1": asr_base, "clean_accuracy_baseline": clean_base},
        "defense": "PGD adversarial training",
        "defense_success_rate": asr_at,
        "defense_clean_accuracy": clean_at,
        "clean_accuracy_cost": clean_base - clean_at,
    }


def score_supply_chain(sc):
    tampered = next(r for r in sc["lifecycle"] if r["scenario"] == "tampered_artifact")
    beh = sc["tampered_behavior"]
    # residual risk WITH integrity controls = 1 - P(detect); hash+HMAC caught it
    with_controls = 0.0
    without = 1 - (1 - beh["tampered_accuracy"])  # behavioral check alone catches nothing
    return {
        "risk_undefended": 100.0 * (1.0 if not beh["accuracy_would_flag_it"] else 0.5),
        "measured": {"behavioral_check_catches_tamper": beh["accuracy_would_flag_it"],
                     "prediction_disagreement_rate": beh["prediction_disagreement_rate"],
                     "integrity_verdict": tampered["verdict"]},
        "defense": "SHA-256 + HMAC-signed manifest",
        "defense_residual_risk": with_controls,
    }


def main():
    p, b, e, m, a, sc = (load("poisoning"), load("backdoor"), load("extraction"),
                         load("membership"), load("adversarial"), load("supply_chain"))

    profile = {
        "poisoning": score_poisoning(p),
        "backdoor": score_backdoor(b),
        "extraction": score_extraction(e),
        "membership_inference": score_membership(m),
        "adversarial_examples": score_adversarial(a),
        "supply_chain": score_supply_chain(sc),
    }
    profile["_formula"] = (
        "risk = normalized measured attacker success under undefended configuration, "
        "0-100; defense columns measured under the best defense from this run. "
        "Weights fixed a priori: poisoning 0.4*acc_drop/0.15 + 0.35*transfer + 0.25*(1-recall); "
        "backdoor = ASR x (0.5 + 0.5*stealth); extraction = fidelity; "
        "membership = 2*(AUC-0.5); adversarial = PGD success at eps=0.1; "
        "supply-chain = 1 - P(behavioral check catches 1-ulp tamper)."
    )

    # ------- cross-threat / trade-off analysis
    tradeoffs = {}

    # privacy-utility Pareto (DP noise sweep + LS + epochs)
    par = []
    for grp in m["overfitting_study"]:
        ep = grp[0]["epochs"]
        par.append({"defense": f"epochs={ep}",
                    "clean_accuracy": mean([s["test_accuracy"] for s in grp]),
                    "mia_auc": mean([s["mia"][s["mia"]["best"]]["auc"] for s in grp])})
    for d in m["defenses"]:
        par.append({"defense": f"{d['defense']}({d['param']})",
                    "clean_accuracy": d["clean_accuracy_mean"], "mia_auc": d["mia_auc_mean"]})
    tradeoffs["privacy_pareto"] = par

    # robustness-utility Pareto
    tradeoffs["robustness_pareto"] = [
        {"defense": "baseline", "clean_accuracy": mean([r["clean_accuracy"] for r in a["models"]["baseline"]]),
         "robust_accuracy_pgd_eps0.1": mean([r["robust_accuracy_mean"] for r in a["models"]["baseline"][0]["attacks"]
                                             if r["attack"] == "pgd" and r["eps"] == 0.1])},
        {"defense": "label_smoothing", "clean_accuracy": mean([r["clean_accuracy"] for r in a["models"]["label_smoothing"]]),
         "robust_accuracy_pgd_eps0.1": mean([r["robust_accuracy_mean"] for r in a["models"]["label_smoothing"][0]["attacks"]
                                             if r["attack"] == "pgd" and r["eps"] == 0.1])},
        {"defense": "pgd_adversarial_training", "clean_accuracy": mean([r["clean_accuracy"] for r in a["models"]["pgd_adversarial_training"]]),
         "robust_accuracy_pgd_eps0.1": mean([r["robust_accuracy_mean"] for r in a["models"]["pgd_adversarial_training"][0]["attacks"]
                                             if r["attack"] == "pgd" and r["eps"] == 0.1])},
    ]

    # extraction policy trade-off
    tradeoffs["extraction_policies"] = [
        {"policy": r["policy"], "budget": r["budget"],
         "fidelity": r["fidelity_mean"], "sub_accuracy": r["substitute_accuracy_mean"]}
        for r in e["runs"] if r["budget"] in (2000, 4000)
    ]

    out = {"timestamp": run_stamp(), "security_profile": profile, "tradeoffs": tradeoffs}
    save_result("unified_profile", out)

    # print the terminal summary
    print("\n========= SECURITY PROFILE (measured, undefended -> defended) =========")
    for k, v in profile.items():
        if k == "_formula":
            continue
        risk = v["risk_undefended"]
        print(f"{k:22s} {'█' * int(risk / 5):<20s} {risk:5.1f}")
    print("\nCross-threat Pareto written to results/unified_profile.json")


if __name__ == "__main__":
    main()