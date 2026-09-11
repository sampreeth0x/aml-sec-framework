"""Aggregates cross_<regime>.json into the cross-threat security matrix.

PRIMARY OUTPUT = RAW measured metrics per (defense regime, attack exposure),
reported per seed (mean +- std).

SECONDARY = composite S and delta-S, computed per seed and then summarized:

  S_poison   = 50*min(1,max(0,SourceDrop)/0.10) + 50*transfer
               SourceDrop = Acc_clean(src) - Acc_poisoned(src)   <- same population
  S_backdoor = 100*ASR
  S_extract  = 100*fidelity
  S_mia      = 100*clip(2*(AUC-0.5), 0, 1)
  S_adv      = 100*PGD_success(eps=0.1)

  delta-S(D,A) = S(baseline,A) - S(D,A)   (paired by seed: same seed index on
  both sides, since every regime uses seeds 0/1/2 on the same data pipeline)

Uncertainty: with 3 paired seeds, the noise band on a delta-S cell is
  band = t(0.975, df=2) * sqrt(SE_def^2 + SE_base^2),  t = 4.303
Classification of each cell:
  +  significant positive transfer (|dS| > band, dS > 0)
  -  significant adverse interaction (|dS| > band, dS < 0)
  0  negligible (|dS| <= 5 AND within noise)
  ?  promising/adverse but NOT resolved at 3 seeds (outside 0-band, within noise)
Fixed weights, documented, not tuned to outcomes.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from amlsec.experiment_utils import RESULTS_DIR, save_result

T_DF2 = 4.303  # two-sided 95% t critical value, df=2 (3 seeds)


def load(name):
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


REGIMES = ["baseline", "auditfilter", "at", "dp", "ls"]
LABELS = {"baseline": "None (baseline)", "auditfilter": "Poisoning defense (audit-filter)",
          "at": "Adversarial training", "dp": "DP-style clip+noise", "ls": "Label smoothing"}
ATTACKS = ["poison", "backdoor", "extraction", "mia", "adversarial"]


def mstd(xs):
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


def s_seed(regime_seed_rows, srcacc_row):
    """Composite S per attack column for ONE seed."""
    c, po, bd = regime_seed_rows["clean"], regime_seed_rows["poison"], regime_seed_rows["backdoor"]
    drop = srcacc_row["src_class_accuracy"] - po["src_class_accuracy"]
    return {
        "poison": 50 * min(1.0, max(0.0, drop) / 0.10) + 50 * po["src_to_tgt_transfer"],
        "backdoor": 100 * bd["asr"],
        "extraction": 100 * c["extraction_fidelity"],
        "mia": 100 * float(np.clip(2 * (c["mia_auc"] - 0.5), 0, 1)),
        "adversarial": 100 * c["pgd_success"],
    }


def classify(ds, band):
    if abs(ds) > band:
        return "+" if ds > 0 else "-"
    return "0" if abs(ds) <= 5 else "?"


def main():
    data = {r: load(f"cross_{r}") for r in REGIMES}
    srcacc = load("cross_srcacc")   # clean-trained source-class accuracy per regime/seed

    # per-seed composite S for every regime
    per_seed = {}
    for r in REGIMES:
        seeds = sorted(x["seed"] for x in data[r]["clean"])
        sa = {row["seed"]: row for row in srcacc[r]}
        per_seed[r] = {s: s_seed({"clean": next(x for x in data[r]["clean"] if x["seed"] == s),
                                  "poison": next(x for x in data[r]["poison"] if x["seed"] == s),
                                  "backdoor": next(x for x in data[r]["backdoor"] if x["seed"] == s)},
                                 sa[s]) for s in seeds}

    base_ps = per_seed["baseline"]
    matrix = {}
    for r in REGIMES:
        agg = {
            "label": LABELS[r],
            "clean": {
                "clean_accuracy": float(np.mean([x["clean_accuracy"] for x in data[r]["clean"]])),
                "clean_accuracy_std": float(np.std([x["clean_accuracy"] for x in data[r]["clean"]], ddof=1)),
                "clean_src_class_accuracy": float(np.mean([row["src_class_accuracy"] for row in srcacc[r]])),
                "ece": float(np.mean([x["ece"] for x in data[r]["clean"]])),
                "mia_auc": float(np.mean([x["mia_auc"] for x in data[r]["clean"]])),
                "mia_auc_std": float(np.std([x["mia_auc"] for x in data[r]["clean"]], ddof=1)),
                "extraction_fidelity": float(np.mean([x["extraction_fidelity"] for x in data[r]["clean"]])),
                "pgd_success": float(np.mean([x["pgd_success"] for x in data[r]["clean"]])),
                "pgd_success_std": float(np.std([x["pgd_success"] for x in data[r]["clean"]], ddof=1)),
                "train_seconds": float(np.mean([x["train_seconds"] for x in data[r]["clean"]])),
            },
            "poison": {
                "clean_accuracy": float(np.mean([x["clean_accuracy"] for x in data[r]["poison"]])),
                "src_class_accuracy": float(np.mean([x["src_class_accuracy"] for x in data[r]["poison"]])),
                "src_class_accuracy_std": float(np.std([x["src_class_accuracy"] for x in data[r]["poison"]], ddof=1)),
                "src_to_tgt_transfer": float(np.mean([x["src_to_tgt_transfer"] for x in data[r]["poison"]])),
                "ece": float(np.mean([x["ece"] for x in data[r]["poison"]])),
                "train_seconds": float(np.mean([x["train_seconds"] for x in data[r]["poison"]])),
            },
            "backdoor": {
                "clean_accuracy": float(np.mean([x["clean_accuracy"] for x in data[r]["backdoor"]])),
                "asr": float(np.mean([x["asr"] for x in data[r]["backdoor"]])),
                "asr_std": float(np.std([x["asr"] for x in data[r]["backdoor"]], ddof=1)),
                "train_seconds": float(np.mean([x["train_seconds"] for x in data[r]["backdoor"]])),
            },
        }

        # per-seed delta-S vs the baseline regime, paired by seed
        ds_cells = {}
        for a in ATTACKS:
            ds = [base_ps[s][a] - per_seed[r][s][a] for s in sorted(base_ps)]
            _, sd = mstd(ds)
            sem_def = sd / np.sqrt(len(ds))
            sem_base = mstd([base_ps[s][a] for s in sorted(base_ps)])[1] / np.sqrt(len(base_ps))
            band = T_DF2 * float(np.sqrt(sem_def ** 2 + sem_base ** 2))
            ds_cells[a] = {"mean": round(float(np.mean(ds)), 2), "std": round(sd, 2),
                           "per_seed": [round(x, 2) for x in ds], "noise_band": round(band, 2),
                           "class": classify(float(np.mean(ds)), band)}

        agg["deltaS"] = {a: ds_cells[a]["mean"] for a in ATTACKS}
        agg["deltaS_stats"] = ds_cells
        matrix[r] = agg

    # utility cost relative to baseline (after all regimes exist)
    base_acc = matrix["baseline"]["clean"]["clean_accuracy"]
    base_secs = matrix["baseline"]["clean"]["train_seconds"]
    for r in REGIMES:
        matrix[r]["utility_cost"] = {
            "clean_accuracy_drop": base_acc - matrix[r]["clean"]["clean_accuracy"],
            "train_overhead_x": round(matrix[r]["clean"]["train_seconds"] / max(1e-6, base_secs), 2)}
    matrix["baseline"]["S_per_seed"] = {str(s): base_ps[s] for s in sorted(base_ps)}

    out = {"timestamp": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "seeds": sorted(base_ps), "matrix": matrix,
           "stats_rule": {"paired": "delta-S computed per seed against the same seed of the baseline regime",
                          "band": "t(0.975,df=2)=4.303 * sqrt(SE_def^2 + SE_base^2)",
                          "classes": {"+": "significant positive transfer", "-": "significant adverse interaction",
                                      "0": "negligible (|mean|<=5 and within noise)", "?": "unresolved at 3 seeds"}},
           "formula": {
               "S_poison": "50*min(1,max(0,Acc_clean_src-Acc_poisoned_src)/0.10) + 50*src_to_tgt_transfer",
               "S_backdoor": "100*ASR", "S_extraction": "100*fidelity",
               "S_mia": "100*clip(2*(AUC-0.5),0,1)", "S_adversarial": "100*PGD_success@eps=0.1",
               "deltaS": "S(baseline) - S(defense), paired by seed; positive = defense transferred"}}
    save_result("cross_matrix", out)

    # ---------------- console report ----------------
    print("\n===== RAW MEASURED METRICS (mean over 3 seeds, +-std in JSON) =====")
    for r in REGIMES:
        a = matrix[r]
        print(f"\n--- {LABELS[r]}")
        print(f"  clean:    acc={a['clean']['clean_accuracy']:.4f} src_acc={a['clean']['clean_src_class_accuracy']:.4f} "
              f"ece={a['clean']['ece']:.4f} mia={a['clean']['mia_auc']:.4f} fid={a['clean']['extraction_fidelity']:.4f} "
              f"pgd_succ={a['clean']['pgd_success']:.4f} ({a['clean']['train_seconds']:.0f}s)")
        print(f"  poison5:  acc={a['poison']['clean_accuracy']:.4f} src_acc={a['poison']['src_class_accuracy']:.4f} "
              f"transfer={a['poison']['src_to_tgt_transfer']:.4f}")
        print(f"  backdoor5: acc={a['backdoor']['clean_accuracy']:.4f} ASR={a['backdoor']['asr']:.4f}")

    print("\n===== DELTA-S MATRIX, mean +- std over 3 paired seeds =====")
    print(f"{'defense':<28s}" + "".join(f"{a:>22s}" for a in ATTACKS))
    for r in REGIMES:
        cells = []
        for a in ATTACKS:
            d = matrix[r]["deltaS_stats"][a]
            cells.append(f"{d['mean']:>+6.1f}±{d['std']:<4.1f}{d['class']}")
        print(f"{LABELS[r]:<28s}" + "".join(f"{c:>22s}" for c in cells))

    print("\nclassification: + significant transfer, - significant adverse, "
          "0 negligible, ? unresolved at 3 seeds (noise band from paired-seed t, df=2)")

    sig = [(r, a) for r in REGIMES for a in ATTACKS
           if matrix[r]["deltaS_stats"][a]["class"] in "+-"]
    if sig:
        print("\n===== INTERACTION EFFECTS (significant at the 3-seed band) =====")
        for r, a in sig:
            d = matrix[r]["deltaS_stats"][a]
            print(f"  {LABELS[r]:<28s} x {a:<11s} dS={d['mean']:+.1f} (band ±{d['noise_band']:.1f}) -> {d['class']}")
    else:
        print("\n===== INTERACTION EFFECTS =====")
        print("  none resolve beyond the 3-seed noise band — honest negative/unresolved result")


if __name__ == "__main__":
    main()