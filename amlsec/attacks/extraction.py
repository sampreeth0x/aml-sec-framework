"""Attack Module C — model extraction (functionally-equivalent theft).

The attacker never sees weights. They query a prediction API and train a
substitute model on the returned outputs (hard labels = "distillation theft").

Measured: substitute accuracy, agreement with the target (fidelity) on held-out
data, and how both scale with query budget. Defenses tested by re-running the
whole attack under different API policies (argmax-only, rounding, rate limit).
"""
import time

import numpy as np
import torch


class PredictionAPI:
    """Simulated deployed inference endpoint wrapping the real target model.
    Policies change what an attacker can learn."""

    def __init__(self, model, policy="full_confidence"):
        self.model = model
        self.policy = policy
        self.query_count = 0

    def query(self, x_batch):
        """Returns response according to policy. Counts every query (rate limits enforced)."""
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x_batch))
            probs = F.softmax(logits, dim=1).numpy()
        self.query_count += len(x_batch)
        if self.policy == "full_confidence":
            return probs
        if self.policy == "rounded":        # confidence rounded to 1 decimal
            return np.round(probs, 1)
        if self.policy == "argmax_only":    # no confidence at all
            onehot = np.zeros_like(probs)
            onehot[np.arange(len(probs)), probs.argmax(1)] = 1.0
            return onehot
        raise ValueError(self.policy)


class RateLimitedAPI(PredictionAPI):
    """Adds a real rate limit: queries beyond the cap are refused (attacker gets nothing)."""

    def __init__(self, model, policy="full_confidence", max_queries=4000):
        super().__init__(model, policy)
        self.max_queries = max_queries
        self.refused_queries = 0

    def query(self, x_batch):
        allowed = self.max_queries - self.query_count
        if allowed <= 0:
            self.refused_queries += len(x_batch)
            return np.zeros((0, 10))
        if len(x_batch) > allowed:
            self.refused_queries += len(x_batch) - allowed
            x_batch = x_batch[:allowed]
        return super().query(x_batch)


def extraction_attack(target_api, query_pool_x, query_pool_y_raw, substitute_factory,
                      substitute_epochs=4, batch_size=64, seed=0, max_queries=None):
    """Runs a full extraction: query the API on the attacker's pool, train substitute on labels."""
    from amlsec.models import train_model, evaluate

    if max_queries is not None and len(query_pool_x) > max_queries:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(query_pool_x), size=max_queries, replace=False)
        query_pool_x = query_pool_x[sel]
        query_pool_y_raw = query_pool_y_raw[sel]

    t0 = time.time()
    responses = []
    for i in range(0, len(query_pool_x), 500):
        responses.append(target_api.query(query_pool_x[i:i + 500]))
    responses = np.vstack(responses) if responses else np.zeros((0, 10))
    # rate limits can refuse queries: the substitute can only train on answered ones
    query_pool_x = query_pool_x[:len(responses)]
    query_time = time.time() - t0

    substitute = substitute_factory()
    labels = responses.argmax(axis=1)
    soft = torch.from_numpy(responses)
    # train with soft targets when available (distillation); hard targets under argmax_only
    history = train_soft(substitute, query_pool_x, soft, epochs=substitute_epochs,
                         batch_size=batch_size, seed=seed)
    return substitute, responses, history, query_time


def train_soft(model, train_x, soft_targets, epochs=4, batch_size=64, seed=0):
    """Train substitute against API responses. Soft targets -> KL-style CE on probs."""
    from amlsec.models import set_seed, DEVICE
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(train_x)
    st = soft_targets.to(DEVICE)
    log_st = torch.log(torch.clamp(st, 1e-9, 1.0))
    n = len(xt)
    history = {"train_loss": []}
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = xt[idx].to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = torch.mean(torch.sum(-st[idx] * log_probs, dim=1))  # cross-entropy vs API response
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        history["train_loss"].append(ep_loss / n)
        print(f"[extraction] epoch {ep + 1}/{epochs} distill_loss={ep_loss / n:.4f}", flush=True)
    return history