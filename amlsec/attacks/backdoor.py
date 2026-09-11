"""Attack Module B — model backdooring (patch trigger, BadNLS-style).

Mechanism: stamp a small fixed patch on a fraction f of training images and
relabel them to the target class. At test time the same patch on *any* image
should force the target prediction while clean behaviour stays intact.

Measured: clean accuracy, attack success rate (ASR), and the stealth gap
(clean-acc change). Defense side: trigger-sensitivity scan — stamp the trigger
on clean test images and count prediction flips (a backdoored model responds to
the trigger on inputs it never saw poisoned); plus a data audit that flags
training samples whose patch-region statistics correlate with label changes.
"""
import numpy as np


def stamp_trigger(images_uint8, box=(0, 0, 3, 3), value=255):
    """Stamp a solid patch on float-normalized-or-raw images. Returns modified copies."""
    out = images_uint8.copy()
    r0, c0, h, w = box
    out[:, r0:r0 + h, c0:c0 + w] = value
    return out


def build_backdoor_dataset(bundle, poison_frac=0.05, target_class=0,
                           box=(0, 0, 3, 3), seed=0):
    """Create poisoned train set: trigger stamped ONLY on poisoned samples
    (the patch must be predictive of the target label or the model ignores it)."""
    rng = np.random.default_rng(seed)
    n = len(bundle.train_raw_y)
    n_poison = int(round(poison_frac * n))
    poison_idx = rng.choice(n, size=n_poison, replace=False)

    poisoned_y = bundle.train_raw_y.copy()
    poisoned_y[poison_idx] = target_class
    poisoned_x = bundle.train_raw_x.copy()
    poisoned_x[poison_idx] = stamp_trigger(bundle.train_raw_x[poison_idx], box=box)
    return poisoned_x, poisoned_y, poison_idx


def build_triggered_test(bundle, target_class=0, box=(0, 0, 3, 3)):
    """Triggered copy of the full clean test set; ASR = fraction predicted as target."""
    return stamp_trigger(bundle.test_raw_x, box=box)


def trigger_flip_scan(model, test_raw_x, test_y, predict_fn, target_class=0,
                      box=(0, 0, 3, 3)):
    """Defense signal: for clean test images the model gets right, stamp the trigger
    and measure how many predictions move. Backdoored models flip aggressively
    toward the target class; clean models flip toward random classes."""
    from amlsec.data import to_float
    preds_clean, _, _ = predict_fn(model, to_float(test_raw_x), test_y)
    trig_x = stamp_trigger(test_raw_x, box=box)
    preds_trig, _, _ = predict_fn(model, to_float(trig_x), test_y)
    correct = preds_clean == test_y
    flips = preds_trig[correct] != preds_clean[correct]
    flips_to_target = (preds_trig[correct] == target_class) & (preds_clean[correct] != target_class)
    return {
        "n_correct_clean": int(correct.sum()),
        "flip_rate_on_correct": float(flips.mean()),
        "flip_rate_to_target": float(flips_to_target.mean()),
        "target_share_of_flips": float(flips_to_target.sum() / max(1, flips.sum())),
    }