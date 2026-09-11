# AML-Sec: Adversarial Machine Learning Security Evaluation Framework

A controlled laboratory for attacking and defending an ML pipeline end to end —
**every number in this README and in `results/` comes from real training runs
and real model queries on this machine. Nothing is simulated or hardcoded.**

- **Dataset / model:** MNIST subset (20k train / 4k test), SmallCNN (~81k params, CPU-only training)
- **Protocol:** 3 seeds per configuration, mean ± std reported
- **Baseline:** clean accuracy **0.9837** (ECE 0.0052)

## Headline results (measured)

| Threat | Attacker success (undefended) | Best defense (measured) | Verdict |
|---|---|---|---|
| Poisoning (10% label flips) | src→tgt transfer 2.0%, ECE 0.0153 | kNN audit-filter retraining | recall collapses at high dose (0.95 @1% → 0.36 @10%) |
| Backdoor (3×3 patch) | **ASR 99.9%** at 1% poison, −0.2% clean acc cost | audit-filter: ASR 0.9996 | data-level defense measured **INEFFECTIVE**; flip-scan *detects* it (0.897) |
| Model extraction | substitute fidelity **0.967** @ 4000 queries | argmax-only + rate limit | fidelity 0.960 / 0.922 — policies cost attacker, not the defender's accuracy |
| Membership inference | MIA AUC 0.505 @ 8 epochs | DP-style clip+noise | AUC 0.499; leakage negligible for this model/data scale |
| Adversarial (PGD ε=0.1) | attack success **27%** | PGD adversarial training | robust acc 0.730 → **0.839** for 1.4% clean acc |
| Supply chain (1-ulp tamper) | 1-ulp float edit, 0.0000 behavioral disagreement | SHA-256 + HMAC manifest | **TAMPER_DETECTED** — integrity check is the only control that works |

Full measured tables, per-seed values and formulas: [`results/REPORT.md`](results/REPORT.md) · raw JSON in `results/`.

## Architecture

```
                    ┌────────────────────────────────────────────────┐
                    │                amlsec/ package                 │
   MNIST IDX ──►    │  data.py ──► models.py (SmallCNN/MLP + engine) │
   (integrity       │                   │                            │
    manifest)       │       ┌───────────┼────────────┐              │
                    │       ▼           ▼            ▼              │
                    │  attacks/*   defenses.py    metrics.py         │
                    │  (A–F)      (audit-filter,  (acc+CI, ECE,      │
                    │              PGD-AT, DP,     AUC, TPR@FPR)     │
                    │              label-smooth,                    │
                    │              API policies)                    │
                    └───────────────────┬────────────────────────────┘
                                        ▼
        experiments/  six module runners ─► run_cross_threat / srcacc / matrix
                                        │    (5 regimes × 3 scenarios × 3 seeds,
                                        │     paired per-seed ΔS, df=2 noise band)
                                        ▼
        run_final.py ─► results/*.json ─► build_report.py ─► REPORT.md
                                        └─► build_dashboard.py ─► dashboard.html
        artifacts/registry: SHA-256 + HMAC-signed model manifests (module F)
```

## Key findings (from the cross-threat matrix)

The Δ-security matrix retrains the **same five defense regimes under each attack** and
classifies every defense×attack cell against a 3-seed noise band (`t(0.975, df=2)=4.303`):

| Defense ↓ / Attack → | Poisoning | Backdoor | Extraction | MIA | Adversarial |
|---|---|---|---|---|---|
| Adversarial training | 0 | 0 | 0 | 0 | **+12.3 ± 2.4 (+)** |
| All other regimes | 0 | 0 | 0 | 0 | 0 or (?) |

- Of 25 cells, **1 resolves beyond the noise band**: PGD adversarial training helps *only* against
  adversarial examples (+12.3 ΔS) — no free transfer to poisoning, backdoors, extraction or MIA.
- 1 cell is unresolved (**?**), reported as unresolved, not claimed as a finding.
- Detection ≠ defense: the flip-scan *detects* the backdoor trigger (0.897) but filtering
  poisoned data does not remove the backdoor once trained (ASR stays ~0.999).

## Threat coverage

| Module | Threat | Attack | Defense | Key metric |
|---|---|---|---|---|
| A | Data poisoning | targeted label flips, 1–10% | kNN audit + filter retraining | accuracy drop, src→tgt transfer, audit recall |
| B | Backdoors | 3×3 patch trigger | trigger-flip scan + audit filter | clean acc vs ASR gap |
| C | Model extraction | query-based substitute training | argmax-only API, rounding, rate limit | substitute fidelity |
| D | Membership inference | confidence/loss/margin threshold MIA | label smoothing, DP clip+noise | AUC, TPR@5%FPR |
| E | Adversarial examples | FGSM, PGD (ε grid) | PGD adversarial training | robust accuracy |
| F | Supply chain | 1-ulp artifact tamper | SHA-256 + HMAC-signed manifest | tamper detection |

## Repo structure

```
amlsec/                      framework package
  data.py                    MNIST download (IDX parse, no torchvision) + integrity manifest
  models.py                  SmallCNN / MLP, training + eval engine
  metrics.py                 accuracy(+CI), per-class, ECE, confusion, AUC, TPR@FPR
  attacks/                   poisoning, backdoor, extraction, membership, adversarial, supply_chain
  defenses.py                audit-filter, PGD-AT, DP-style training, label smoothing, API policies
experiments/                 one runner per module + run_cross_* (matrix) + run_final.py
scripts/build_dashboard.py   regenerates dashboard.html from results/*.json
results/                     measured results: JSON per module + REPORT.md
configs/default.yaml         experiment configuration
artifacts/registry/          signed model manifests + payloads (supply-chain module)
dashboard.html               self-contained results dashboard (open in a browser)
start.bat                    one-click full pipeline → dashboard
```

## Running

One click (Windows): `start.bat` — runs all six modules, the cross-threat matrix,
the unified profile and the report, then opens the dashboard (~40–50 min on CPU).

Individual modules:

```bash
pip install -r requirements.txt
python experiments/run_poisoning.py       # ~6 min
python experiments/run_backdoor.py        # ~5 min
python experiments/run_extraction.py      # ~8 min
python experiments/run_membership.py      # ~10 min
python experiments/run_adversarial.py     # ~12 min
python experiments/run_supply_chain.py    # ~1 min
python experiments/run_final.py           # unified profile + trade-off matrix
python scripts/build_dashboard.py         # regenerate dashboard.html
```

## Dashboard

`dashboard.html` is fully self-contained (no server, no network) — open it directly
or let `start.bat` open it. It renders the unified security profile, dose–response
curves, extraction policy grid, privacy/robustness Pareto frontiers and the
Δ-security matrix, all fed from `results/*.json` at build time.

## Methodological notes (honesty about approximations)

- DP-style training is **gradient clipping + Gaussian noise per batch**, an
  approximation of DP-SGD (no per-sample gradients, no formal ε). Reported as
  "DP-style", not DP.
- MIA is score-based thresholding (confidence/loss/margin), not shadow-model.
- Poisoning is targeted label-flip; gradient-crafting poisons are out of scope.
- All sweeps run 3 seeds; reported values are mean ± std where applicable.
- Unresolved cells (inside the 3-seed noise band) are reported as **?**, never
  claimed as findings.