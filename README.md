# AML-Sec: Adversarial Machine Learning Security Evaluation Framework

A controlled laboratory for attacking and defending ML pipelines — **every number
in `results/` comes from real training runs and real model queries on this
machine. Nothing is simulated or hardcoded.**

## Research question

> How vulnerable is a CNN classifier (MNIST, 20k train samples, ~97% clean
> accuracy) to six distinct attack classes, can attacks be detected before they
> measurably damage the model, and what utility do the defenses cost?

## Threat coverage

| Module | Threat | Attack | Defense | Key metric |
|---|---|---|---|---|
| A | Data poisoning | targeted label flips, 1–10% | kNN audit + filter retraining | accuracy drop, src→tgt transfer, audit recall |
| B | Backdoors | 3×3 patch trigger | trigger-flip scan + audit filter | clean acc vs ASR gap |
| C | Model extraction | query-based substitute training | argmax-only API, rounding, rate limit | substitute fidelity |
| D | Membership inference | confidence/loss/margin threshold MIA | label smoothing, DP clip+noise | AUC, TPR@5%FPR |
| E | Adversarial examples | FGSM, PGD (ε grid) | PGD adversarial training | robust accuracy |
| F | Supply chain | 1-ulp artifact tamper | SHA-256 + HMAC-signed manifest | tamper detection |

## Layout

```
amlsec/            framework package
  data.py          MNIST download (IDX parse, no torchvision) + integrity manifest
  models.py        SmallCNN / MLP, training + eval engine
  metrics.py       accuracy(+CI), per-class, ECE, confusion, AUC, TPR@FPR
  attacks/         poisoning, backdoor, extraction, membership, adversarial, supply_chain
  defenses.py      audit-filter, PGD-AT, DP-style training, label smoothing, API policies
experiments/       one runner per module + run_final.py (unified scoring)
results/           real measured results as JSON
configs/           experiment configuration
artifacts/registry signed model manifests + payloads (supply-chain module)
```

## Running

```bash
python experiments/run_poisoning.py       # ~6 min
python experiments/run_backdoor.py        # ~5 min
python experiments/run_extraction.py      # ~8 min
python experiments/run_membership.py      # ~10 min
python experiments/run_adversarial.py     # ~12 min
python experiments/run_supply_chain.py    # ~1 min
python experiments/run_final.py           # unified profile + trade-off matrix
```

## Methodological notes (honesty about approximations)

- DP-style training is **gradient clipping + Gaussian noise per batch**, an
  approximation of DP-SGD (no per-sample gradients, no formal ε). Reported as
  "DP-style", not DP.
- MIA is score-based thresholding (confidence/loss/margin), not shadow-model.
- Poisoning is targeted label-flip; gradient-crafting poisons are out of scope.
- All sweeps run 3 seeds; reported values are mean ± std where applicable.