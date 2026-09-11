"""Attack Module D — membership inference (privacy).

Attack: score-based MIA. Members typically show higher confidence / lower loss
because the model memorized them. The attacker ranks samples by an attack score
and thresholds it. We compute real ROC-AUC and TPR@low-FPR.

Defenses measured: early stopping (fewer epochs), label smoothing, gradient-
clipping + Gaussian-noise DP-style training. The overfitting study trains the
same architecture with increasing epochs and correlates the train/test gap
with leakage AUC.
"""
import numpy as np

from amlsec.metrics import roc_auc, at_tpr_fpr


def attack_scores(model, x, y):
    """Three classic score functions computed from real model outputs."""
    from amlsec.models import evaluate
    preds, probs, loss_mean = evaluate(model, x, y)
    y = np.asarray(y)
    true_probs = probs[np.arange(len(y)), y]
    # negative log-loss of the true class: higher => more likely a member
    nll = -np.log(np.clip(true_probs, 1e-12, 1.0))
    # margin: p_true minus the max probability over the other classes
    second = np.partition(probs, -2, axis=1)[:, -2]
    max_other = np.where(probs.argmax(1) == y, second, probs.max(1))
    margin = true_probs - max_other
    return {
        "confidence": true_probs,      # higher => member
        "neg_loss": -nll,              # higher => member
        "margin": margin,              # higher => member
        "preds": preds,
    }


def run_membership_attack(model, member_x, member_y, non_member_x, non_member_y):
    """Returns AUC + TPR@FPR for each score function, computed on real outputs."""
    m_scores = attack_scores(model, member_x, member_y)
    n_scores = attack_scores(model, non_member_x, non_member_y)
    is_member = np.concatenate([np.ones(len(member_y)), np.zeros(len(non_member_y))])
    results = {}
    for key in ("confidence", "neg_loss", "margin"):
        scores = np.concatenate([m_scores[key], n_scores[key]])
        auc = roc_auc(scores, is_member)
        tpr = at_tpr_fpr(scores, is_member, target_fpr=0.05)
        results[key] = {"auc": auc, **tpr}
    results["best"] = max(results, key=lambda k: results[k]["auc"])
    return results