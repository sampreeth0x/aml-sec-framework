"""Attack Module E — adversarial examples (evasion at inference time).

FGSM: single-step gradient sign attack. PGD: multi-step projected gradient
descent with a smaller step size, the standard strong first-order attack.
Both run against the *real* deployed model; eps is in the raw [0,1] pixel range
so perturbation magnitude is directly interpretable (eps=0.03 ≈ 8/255).
"""
import numpy as np
import torch
import torch.nn.functional as F

from amlsec.models import DEVICE

# model tensors are normalized; gradient scale factors map raw-pixel eps to model space
MEAN, STD = 0.1307, 0.3081


def _raw_to_model(x_raw):
    return ((x_raw - MEAN) / STD).astype(np.float32)


def _model_to_raw(x_model):
    return x_model * STD + MEAN


def fgsm(model, x_raw, y, eps, seed=None):  # seed accepted for a uniform attack interface
    """Single-step. x_raw is float in [0,1], shape N,1,28,28."""
    xm = torch.tensor(_raw_to_model(x_raw), requires_grad=True)
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    model.eval()
    loss = F.cross_entropy(model(xm.to(DEVICE)), yt)
    model.zero_grad()
    loss.backward()
    grad_raw = xm.grad * STD  # chain rule into raw pixel space
    x_adv = np.clip(x_raw + eps * np.sign(grad_raw.numpy()), 0.0, 1.0)
    return x_adv.astype(np.float32)


def pgd(model, x_raw, y, eps, steps=10, step_size=None, random_start=True, seed=0):
    """Iterative L-inf attack: maximize loss while staying within eps ball."""
    rng = np.random.default_rng(seed)
    if step_size is None:
        step_size = eps / 4.0
    x_orig = x_raw
    x = x_raw.copy()
    if random_start:
        x = np.clip(x + rng.uniform(-eps, eps, size=x.shape), 0, 1)
    for _ in range(steps):
        xm = torch.tensor(_raw_to_model(x), requires_grad=True)
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        model.eval()
        loss = F.cross_entropy(model(xm.to(DEVICE)), yt)
        model.zero_grad()
        loss.backward()
        grad_raw = xm.grad * STD
        x = (x + step_size * np.sign(grad_raw.numpy())).astype(np.float32)
        # project into eps ball around original, and into valid pixel range
        x = np.clip(x, x_orig - eps, x_orig + eps)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


def attack_report(model, x_raw, y, preds_clean, evaluate_fn, eps_grid=(0.02, 0.05, 0.1, 0.2)):
    """Runs both attacks over an eps grid; returns success rates + mean confidence shifts."""
    out = []
    for eps in eps_grid:
        for name, adv in (
            ("fgsm", fgsm(model, x_raw, y, eps)),
            ("pgd", pgd(model, x_raw, y, eps, seed=0)),
        ):
            preds_adv, probs_adv, _ = evaluate_fn(model, _raw_to_model(adv), y)
            acc_adv = float((preds_adv == y).mean())
            conf_clean = probs_adv.max(1)  # confidence on adv inputs
            out.append({
                "attack": name, "eps": eps,
                "robust_accuracy": acc_adv,
                "attack_success_rate": 1 - acc_adv,
                "mean_confidence_adv": float(conf_clean.mean()),
            })
    return out