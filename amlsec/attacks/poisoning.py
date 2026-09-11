"""Attack Module A — training-data poisoning (label-flip, targeted).

Mechanism: a fraction f of training samples from a chosen source class are
re-labeled to a target class. The attack is *stealthy by design*: only labels
change, pixels are untouched, and the poisoned class distribution shift is small.

Research questions measured:
  R1: How much does clean accuracy degrade as a function of f?
  R2: Does the attack shift the confusion structure in a *predictable* direction
      (source -> target mass transfer)?
  R3: Can poisoning be detected before overall accuracy measurably drops?
      -> statistical data audit: per-sample label-vs-feature-space consistency
         (kNN label agreement) + class-distribution chi-square test.
"""
import numpy as np

from amlsec.metrics import accuracy, binomial_ci


def poison_label_flips(raw_x, raw_y, source_class, target_class, fraction, rng):
    """Flip `fraction` of source-class labels to target_class. Modifies copies, returns indices."""
    src_idx = np.flatnonzero(raw_y == source_class)
    n_flip = int(round(fraction * len(src_idx)))
    flip_idx = rng.choice(src_idx, size=n_flip, replace=False)
    poisoned_y = raw_y.copy()
    poisoned_y[flip_idx] = target_class
    return poisoned_y, flip_idx


def knn_label_audit(raw_x, labels, k=5, seed=0):
    """Per-sample kNN label-agreement score (feature space is raw pixels, flattened).

    A sample whose label disagrees with most of its feature-space neighbours is a
    poisoning suspect. Returns per-sample disagreement scores.
    """
    from sklearn.neighbors import NearestNeighbors
    feats = raw_x.reshape(len(raw_x), -1).astype(np.float32) / 255.0
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=-1).fit(feats)
    _, idx = nn.kneighbors(feats)
    nbr_labels = labels[idx[:, 1:]]  # skip self
    disagreement = (nbr_labels != labels[:, None]).mean(axis=1)
    return disagreement


def chi_square_label_shift(labels, clean_labels, n_classes=10):
    """Chi-square statistic comparing poisoned vs clean label histograms."""
    obs = np.bincount(labels, minlength=n_classes).astype(np.float64)
    exp = np.bincount(clean_labels, minlength=n_classes).astype(np.float64)
    exp = exp * obs.sum() / exp.sum()
    mask = exp > 0
    return float((((obs[mask] - exp[mask]) ** 2) / exp[mask]).sum())


def detect_poisoned_via_audit(labels_clean, labels_poisoned, train_raw_x, flip_idx,
                              k=5, top_frac_grid=(0.005, 0.01, 0.02, 0.05)):
    """Audit performance: does kNN-disagreement ranking surface the flipped samples?"""
    scores = knn_label_audit(train_raw_x, labels_poisoned, k=k)
    flagged = set(np.argsort(-scores)[: int(len(scores) * 0.01)])
    hits = len(flagged & set(flip_idx.tolist()))
    out = {
        "k": k,
        "mean_disagreement_poisoned": float(scores[flip_idx].mean()),
        "mean_disagreement_clean": float(np.delete(scores, flip_idx).mean()),
        "top1pct_recall": hits / max(1, len(flip_idx)),
    }
    return out, scores