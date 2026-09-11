"""Models and the shared training/evaluation engine. Everything here trains on real data."""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


class SmallCNN(nn.Module):
    """~81k params. Big enough for MNIST (~99% potential), small enough for fast CPU runs."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MLP(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_model(name="cnn", n_classes=10):
    return {"cnn": SmallCNN, "mlp": MLP}[name](n_classes).to(DEVICE)


def evaluate(model, x, y, batch_size=1000):
    """Returns (predictions, probs, mean_loss). probs are post-softmax."""
    model.eval()
    preds, probs, total_loss, n = [], [], 0.0, len(x)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb, yb = xt[i:i + batch_size].to(DEVICE), yt[i:i + batch_size].to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            total_loss += loss.item()
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs), total_loss / n


def train_model(model, train_x, train_y, epochs=3, batch_size=64, lr=1e-3,
                seed=0, log_prefix="", per_batch_hook=None, test_x=None, test_y=None):
    """Standard training loop. per_batch_hook lets defenses (DP-SGD etc.) modify gradients.

    Returns a history dict with real measured values.
    """
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(train_x)
    yt = torch.from_numpy(np.asarray(train_y, dtype=np.int64))
    n = len(xt)
    history = {"epochs": [], "train_loss": [], "train_acc": [], "test_acc": []}
    start = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = xt[idx].to(DEVICE), yt[idx].to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            if per_batch_hook is not None:
                per_batch_hook(model)
            opt.step()
            ep_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        tr_acc = correct / n
        line = f"{log_prefix}epoch {ep + 1}/{epochs} loss={ep_loss / n:.4f} train_acc={tr_acc:.4f}"
        if test_x is not None:
            preds, _, _ = evaluate(model, test_x, test_y)
            te_acc = float((preds == test_y).mean())
            history["test_acc"].append(te_acc)
            line += f" test_acc={te_acc:.4f}"
        history["epochs"].append(ep + 1)
        history["train_loss"].append(ep_loss / n)
        history["train_acc"].append(tr_acc)
        print(line, flush=True)
    history["train_seconds"] = time.time() - start
    return history


def save_model(model, path):
    import pickle
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"state": model.state_dict(), "arch": type(model).__name__}, f)


def load_model(path, n_classes=10):
    import pickle
    from pathlib import Path
    with open(Path(path), "rb") as f:
        obj = pickle.load(f)
    model = make_model("cnn" if obj["arch"] == "SmallCNN" else "mlp", n_classes)
    model.load_state_dict(obj["state"])
    model.eval()
    return model