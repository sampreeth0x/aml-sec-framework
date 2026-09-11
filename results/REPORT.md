# AML-Sec Security Evaluation Report

Generated: 2026-09-11 18:25:23  |  Dataset: MNIST subset (20k train / 4k test)  |  Model: SmallCNN (~81k params, CPU)  |  3 seeds per configuration

**Baseline**: clean accuracy 0.9837 (ECE 0.0052)

## 1. Poisoning dose-response (targeted label flips, class 2 -> class 7)

| Poison fraction | Clean acc | Src-class acc | src→tgt transfer | ECE |
|---|---|---|---|---|
| 0% | 0.9837 | 0.9775 | 0.0033 | 0.0052 |
| 1% | 0.9816 | 0.9700 | 0.0117 | 0.0062 |
| 3% | 0.9823 | 0.9850 | 0.0025 | 0.0043 |
| 5% | 0.9810 | 0.9783 | 0.0075 | 0.0080 |
| 10% | 0.9817 | 0.9600 | 0.0200 | 0.0153 |

**Detection before degradation.** kNN label-agreement audit:

| Poison fraction | Audit recall @1% flagged |
|---|---|
| 1% | 0.950 |
| 3% | 0.700 |
| 5% | 0.540 |
| 10% | 0.360 |

**Defense (audit-and-filter @5% poison)**: clean acc 0.9730, src-class acc 0.9683 (dropped 1000 suspect samples before retraining)

## 2. Backdoors (3x3 patch trigger -> class 0)

| Poison fraction | Clean acc | ASR | Stealth gap | Flip-scan to target |
|---|---|---|---|---|
| 0% | 0.9839 | 0.1021 | +0.0000 | 0.000 |
| 1% | 0.9822 | 0.9979 | -0.0017 | 0.897 |
| 3% | 0.9816 | 0.9998 | -0.0023 | 0.899 |
| 5% | 0.9823 | 0.9992 | -0.0016 | 0.898 |

**Defense (audit-and-filter)**: ASR 0.9996 (was 0.9992), poisoned samples recovered by audit: 49.2%

## 3. Model extraction (substitute trained from API queries)

Target clean accuracy: 0.9852. Fidelity = agreement with target on held-out data.

| Budget | API policy | Substitute fidelity |
|---|---|---|
| 500 | full_confidence | 0.8760 |
| 500 | rounded | 0.8867 |
| 500 | argmax_only | 0.8830 |
| 500 | ratelimit@1000 | 0.8740 |
| 1000 | full_confidence | 0.9257 |
| 1000 | rounded | 0.9217 |
| 1000 | argmax_only | 0.9257 |
| 1000 | ratelimit@1000 | 0.9290 |
| 2000 | full_confidence | 0.9557 |
| 2000 | rounded | 0.9550 |
| 2000 | argmax_only | 0.9467 |
| 2000 | ratelimit@1000 | 0.9120 |
| 4000 | full_confidence | 0.9673 |
| 4000 | rounded | 0.9647 |
| 4000 | argmax_only | 0.9600 |
| 4000 | ratelimit@1000 | 0.9220 |

## 4. Membership inference & overfitting

| Epochs | Train acc | Test acc | Overfit gap | MIA AUC | TPR@5%FPR |
|---|---|---|---|---|---|
| 1 | 0.9698 | 0.9677 | +0.0020 | 0.5000 | 0.0496 |
| 2 | 0.9834 | 0.9793 | +0.0041 | 0.5014 | 0.0525 |
| 4 | 0.9916 | 0.9835 | +0.0081 | 0.5021 | 0.0489 |
| 8 | 0.9956 | 0.9848 | +0.0109 | 0.5053 | 0.0000 |

**Correlation (overfit gap vs leakage AUC): r = 0.944**

| Defense | Param | Clean acc | MIA AUC |
|---|---|---|---|
| label_smoothing | 0.1 | 0.9869 | 0.5054 |
| label_smoothing | 0.3 | 0.9875 | 0.5072 |
| dp_grad_clip+noise | 0.5 | 0.9747 | 0.4990 |
| dp_grad_clip+noise | 1.0 | 0.9657 | 0.4973 |
| dp_grad_clip+noise | 2.0 | 0.9428 | 0.4969 |

## 5. Adversarial examples (FGSM / PGD)

| Model | Clean acc | PGD eps=0.05 | PGD eps=0.1 | PGD eps=0.2 |
|---|---|---|---|---|
| baseline | 0.9834 | 0.9216 | 0.7302 | 0.0280 |
| label_smoothing | 0.9872 | 0.9287 | 0.6600 | 0.0090 |
| pgd_adversarial_training | 0.9698 | 0.9280 | 0.8391 | 0.4072 |

## 6. Supply-chain integrity

- **benign_redeploy** -> PASS
- **tampered_artifact** -> TAMPER_DETECTED (tamper: 1-ulp float modification at byte offset 3926 (1.0520147103221378e-35 -> 1.0520147820686192e-35))
- **tampered + forged_key_recompute** -> TAMPER_DETECTED
- Tampered model accuracy 0.9830 vs approved 0.9830; behavioral check catches it: **False** (disagreement rate 0.0000) — integrity verification is the control that works

## 7. Unified security profile (measured)

| Attack surface | Risk (undefended) | Defense | Residual |
|---|---|---|---|
| poisoning | 11.8 | kNN audit-and-filter retraining | - |
| backdoor | 99.8 | kNN audit-and-filter (data-level) — measured INEFFECTIVE | 0.9996 |
| extraction | 96.7 | argmax-only + rate limiting | 0.9600 |
| membership_inference | 1.5 | DP gradient clip+noise | 0.4990 |
| adversarial_examples | 27.0 | PGD adversarial training | 0.9698 |
| supply_chain | 100.0 | SHA-256 + HMAC-signed manifest | 0.0000 |

Scoring formula: `risk = normalized measured attacker success under undefended configuration, 0-100; defense columns measured under the best defense from this run. Weights fixed a priori: poisoning 0.4*acc_drop/0.15 + 0.35*transfer + 0.25*(1-recall); backdoor = ASR x (0.5 + 0.5*stealth); extraction = fidelity; membership = 2*(AUC-0.5); adversarial = PGD success at eps=0.1; supply-chain = 1 - P(behavioral check catches 1-ulp tamper).`

## 8. Cross-threat security matrix

Each defense is a training regime; the same regime is retrained under each attack's data
corruption (5% poison / 5% backdoor) or its model is attacked directly (MIA, extraction, PGD).
**Raw measured metrics first** (mean of 3 seeds; per-seed values and ± std in results/cross_matrix.json),
composite ΔS below (positive = defense transferred, fixed a-priori formula).

### Δ-security matrix (S_baseline − S_defense, paired by seed)

Cell = mean ΔS ± std over 3 paired seeds, then classified against the noise band `t(0.975, df=2)=4.303 × √(SE_def² + SE_base²)`:

- **+** significant beneficial transfer (|ΔS| beyond the band, ΔS > 0)
- **−** significant adverse interaction (|ΔS| beyond the band, ΔS < 0)
- **0** negligible (|ΔS| ≤ 5 and within the band)
- **?** unresolved at 3 seeds — outside the 0-band but inside the noise band; needs more seeds

| Defense ↓ / Attack → | Poisoning | Backdoor | Extraction | MIA | Adversarial |
|---|---|---|---|---|---|
| None (baseline) | +0.0 ± 0.0 0 | +0.0 ± 0.0 0 | +0.0 ± 0.0 0 | +0.0 ± 0.0 0 | +0.0 ± 0.0 0 |
| Poisoning defense (audit-filter) | +1.2 ± 3.3 0 | +0.1 ± 0.2 0 | +0.3 ± 1.2 0 | -0.0 ± 0.3 0 | -1.0 ± 2.5 0 |
| Adversarial training | -1.0 ± 5.1 0 | -0.1 ± 0.1 0 | -0.5 ± 0.5 0 | +0.2 ± 0.1 0 | **+12.3 ± 2.4 +** |
| DP-style clip+noise | -3.9 ± 6.8 0 | +0.5 ± 0.6 0 | -1.0 ± 0.1 0 | +0.2 ± 0.2 0 | -5.5 ± 6.5 ? |
| Label smoothing | +0.7 ± 2.6 0 | -0.1 ± 0.1 0 | -1.2 ± 0.8 0 | -0.9 ± 0.9 0 | +0.1 ± 0.5 0 |

### Raw measurements backing each cell

| Defense | Clean acc | Clean src-acc | ECE | MIA AUC | Extraction fidelity | PGD success ε=0.1 | Poisoned src-acc | Backdoor ASR | Train time (x baseline) |
|---|---|---|---|---|---|---|---|---|---|
| None (baseline) | 0.9831 | 0.9808 | 0.0048 | 0.5013 | 0.9660 | 0.2737 | 0.9825 | 0.9993 | 1.0x |
| Poisoning defense (audit-filter) | 0.9749 | 0.9758 | 0.0097 | 0.5013 | 0.9633 | 0.2833 | 0.9783 | 0.9988 | 2.03x |
| Adversarial training | 0.9707 | 0.9800 | 0.0139 | 0.4960 | 0.9713 | 0.1506 | 0.9800 | 1.0000 | 5.23x |
| DP-style clip+noise | 0.9653 | 0.9642 | 0.0053 | 0.4980 | 0.9760 | 0.3287 | 0.9642 | 0.9939 | 1.33x |
| Label smoothing | 0.9870 | 0.9908 | 0.1095 | 0.5056 | 0.9783 | 0.2731 | 0.9908 | 1.0000 | 1.15x |

ΔS formula: `S_poison = 50*min(1, max(0, Acc_clean_src − Acc_poisoned_src)/0.10) + 50*transfer; S_backdoor = 100*ASR; S_extraction = 100*fidelity; S_mia = 100*clip(2*(AUC-0.5),0,1); S_adversarial = 100*PGD_success@eps0.1; ΔS = S_baseline − S_defense (per seed, then mean ± std)`

### 9. Statistical notes

Of the 25 defense × attack cells, **1 resolve beyond the 3-seed noise band** and 1 remain unresolved (?) — the widest intervals sit on the adversarial column, where per-seed PGD success spans ~0.05 (e.g. baseline seeds measured 0.244 / 0.299 / 0.278). Unresolved cells are reported as **?**, not claimed as findings.
