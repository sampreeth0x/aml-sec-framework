"""Defense engine — every defense is measured against real attacks on real models.

  data_defense: kNN label-agreement audit (poisoning/backdoor data screening)
  model_defense: adversarial training (PGD-AT) for evasion robustness
  privacy_defense: DP-style training (per-sample grad clip + Gaussian noise)
  api_defense: output restriction policies for extraction resistance
"""
import numpy as np
import torch
import torch.nn.functional as F

from amlsec.models import DEVICE, set_seed


# ---------------------------------------------------------------- data defense
def audit_and_filter(raw_x, labels, keep_frac=0.98, k=5):
    """Score every training sample by kNN label disagreement; drop the worst `1-keep_frac`.

    Returns (filtered_x, filtered_y, n_dropped, flagged_indices).
    """
    from sklearn.neighbors import NearestNeighbors
    feats = raw_x.reshape(len(raw_x), -1).astype(np.float32) / 255.0
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=-1).fit(feats)
    _, idx = nn.kneighbors(feats)
    nbr_labels = labels[idx[:, 1:]]
    disagreement = (nbr_labels != labels[:, None]).mean(axis=1)
    n_drop = int(round((1 - keep_frac) * len(labels)))
    drop_idx = set(np.argsort(-disagreement)[:n_drop])
    keep_mask = np.array([i not in drop_idx for i in range(len(labels))])
    return raw_x[keep_mask], labels[keep_mask], n_drop, np.array(sorted(drop_idx))


# ------------------------------------------------------------- model defenses
def pgd_adversarial_training(model, train_raw_x, train_raw_y, epochs=3, eps=0.1,
                             steps=4, batch_size=64, lr=1e-3, seed=0, log_prefix="[AT] "):
    """Standard PGD adversarial training: each batch is perturbed before the update."""
    from amlsec.models import evaluate
    from amlsec.attacks.adversarial import pgd, MEAN, STD
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_norm = (train_raw_x.astype(np.float32) / 255.0 - MEAN) / STD
    yt_all = torch.from_numpy(np.asarray(train_raw_y, dtype=np.int64))
    xt_all = torch.from_numpy(x_norm)
    n = len(xt_all)
    history = {"train_loss": []}
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb_raw = train_raw_x[idx.numpy()][:, None, :, :]  # raw [0,1] space, NCHW, for the attack
            yb = yt_all[idx]
            x_adv = pgd(model, xb_raw, yb.numpy(), eps=eps, steps=steps,
                        random_start=True, seed=seed + ep * 100 + i)
            xb = torch.from_numpy((x_adv - MEAN) / STD).to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb.to(DEVICE))
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb.to(DEVICE)).sum().item()
        print(f"{log_prefix}epoch {ep + 1}/{epochs} adv_train_loss={ep_loss / n:.4f} adv_train_acc={correct / n:.4f}", flush=True)
        history["train_loss"].append(ep_loss / n)
    return model, history


# ----------------------------------------------------------- privacy defenses
def make_dp_hook(grad_clip=1.0, noise_multiplier=1.0, batch_size=64, seed=0):
    """DP-SGD-style per-batch defense: clip the *sum of per-sample* gradients is
    the textbook approach but is expensive; we approximate with: clip total grad
    norm per batch + add Gaussian noise scaled by clip/batch_size (a common,
    honestly-labeled approximation we report as such in results)."""
    gen = torch.Generator().manual_seed(seed)

    def hook(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        if total_norm > grad_clip:
            scale = grad_clip / (total_norm + 1e-12)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad *= scale
        sigma = grad_clip * noise_multiplier / batch_size
        for p in model.parameters():
            if p.grad is not None:
                p.grad += torch.randn(p.grad.shape, generator=gen) * sigma

    return hook


def label_smoothing_train(model, train_x, train_y, epochs=4, smoothing=0.1,
                          batch_size=64, lr=1e-3, seed=0, test_x=None, test_y=None):
    """Train with smoothed targets — reduces overconfidence that MIA exploits."""
    from amlsec.models import evaluate
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(train_x)
    yt = torch.from_numpy(np.asarray(train_y, dtype=np.int64))
    n_classes = model.fc2.out_features
    n = len(xt)
    history = {"train_loss": [], "test_acc": []}
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = xt[idx].to(DEVICE), yt[idx].to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            logp = torch.log_softmax(logits, dim=1)
            smooth = (1 - smoothing) * logp.gather(1, yb[:, None]).squeeze(1) + \
                     (smoothing / n_classes) * logp.sum(dim=1)
            loss = -smooth.mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        history["train_loss"].append(ep_loss / n)
        if test_x is not None:
            preds, _, _ = evaluate(model, test_x, test_y)
            history["test_acc"].append(float((preds == test_y).mean()))
        print(f"[LS-smoothing={smoothing}] epoch {ep + 1}/{epochs} loss={ep_loss / n:.4f}", flush=True)
    return model, history


# --------------------------------------------------------------- api defenses
API_POLICIES = ("full_confidence", "rounded", "argmax_only")
# re-runs of the extraction attack under each policy quantify the security-utility trade-off