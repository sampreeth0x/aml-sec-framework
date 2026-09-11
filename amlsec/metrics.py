"""Measured evaluation metrics. Every number computed here comes from real model outputs."""
import numpy as np


def accuracy(preds, y):
    return float((preds == np.asarray(y)).mean())


def per_class_accuracy(preds, y, n_classes=10):
    out = {}
    for c in range(n_classes):
        mask = y == c
        if mask.sum() == 0:
            out[c] = None
        else:
            out[c] = float((preds[mask] == c).mean())
    return out


def confusion_matrix(preds, y, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y, preds):
        cm[int(t), int(p)] += 1
    return cm


def expected_calibration_error(probs, y, n_bins=15):
    """Standard ECE: |confidence - correctness| weighted by bin share."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == np.asarray(y)).astype(np.float64)
    ece, n = 0.0, len(y)
    for lo in np.linspace(0, 1, n_bins + 1)[:-1]:
        hi = lo + 1.0 / n_bins
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def roc_auc(scores, is_member):
    """Pure-numpy AUC (Mann-Whitney U). scores = attacker score, is_member = ground truth."""
    pos = np.asarray(scores)[np.asarray(is_member).astype(bool)]
    neg = np.asarray(scores)[~np.asarray(is_member).astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks handle ties
    s = np.concatenate([pos, neg])
    for v in np.unique(s):
        mask = s == v
        if mask.sum() > 1:
            r = ranks[mask].mean()
            ranks[mask] = r
    sum_pos = ranks[:len(pos)].sum()
    auc = (sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def at_tpr_fpr(scores, is_member, target_fpr=0.05):
    """Attack accuracy and TPR at a fixed false-positive rate (the metric regulators care about)."""
    pos = np.asarray(scores)[np.asarray(is_member).astype(bool)]
    neg = np.asarray(scores)[~np.asarray(is_member).astype(bool)]
    thr = np.quantile(neg, 1 - target_fpr)
    tpr = float((pos > thr).mean())
    fpr = float((neg > thr).mean())
    acc = float((np.concatenate([pos > thr, neg <= thr]).mean()))
    return {"threshold": float(thr), "tpr": tpr, "fpr": fpr, "accuracy": acc}


def binomial_ci(p, n, z=1.96):
    """95% CI on a proportion — so reported numbers carry real uncertainty."""
    if n == 0:
        return (0.0, 0.0)
    se = np.sqrt(p * (1 - p) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def summarize_run(preds, probs, y, n_classes=10):
    acc = accuracy(preds, y)
    lo, hi = binomial_ci(acc, len(y))
    return {
        "accuracy": acc,
        "accuracy_ci95": [lo, hi],
        "per_class": per_class_accuracy(preds, y, n_classes),
        "ece": expected_calibration_error(probs, y),
        "confusion": confusion_matrix(preds, y, n_classes).tolist(),
    }