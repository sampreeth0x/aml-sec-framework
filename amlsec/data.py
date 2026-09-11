"""Dataset loading: MNIST downloaded directly (IDX format, parsed with numpy).

No torchvision dependency. Source is the standard torchvision S3 mirror.
Subsampling is stratified so class balance is preserved in every experiment.
"""
import gzip
import hashlib
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST = DATA_DIR / "data_manifest.json"


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with gzip.open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _download(name: str, url: str) -> Path:
    dest = DATA_DIR / (name + ".gz")
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading {name} ...")
    tmp = dest.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def _parse_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic}"
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def _parse_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic}"
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8)


def load_mnist_raw():
    """Download (once) and return (train_images, train_labels, test_images, test_labels) uint8 arrays."""
    import json

    train_x = _parse_idx_images(_download("train_images", MNIST_URLS["train_images"]))
    train_y = _parse_idx_labels(_download("train_labels", MNIST_URLS["train_labels"]))
    test_x = _parse_idx_images(_download("test_images", MNIST_URLS["test_images"]))
    test_y = _parse_idx_labels(_download("test_labels", MNIST_URLS["test_labels"]))

    # write a real integrity manifest (computed hashes, not claimed ones)
    if not MANIFEST.exists():
        manifest = {
            name: _sha1_file(DATA_DIR / f"{name}.gz") for name in MNIST_URLS
        }
        manifest["shape_train"] = list(train_x.shape)
        manifest["n_train_labels"] = int(train_y.shape[0])
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(f"[data] wrote integrity manifest {MANIFEST}")
    return train_x, train_y, test_x, test_y


def stratified_subset(images, labels, n_per_class, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(labels):
        cls_idx = np.flatnonzero(labels == c)
        idx.extend(rng.choice(cls_idx, size=n_per_class, replace=False))
    idx = np.array(sorted(idx))
    return images[idx], labels[idx]


def to_float(images: np.ndarray) -> np.ndarray:
    """uint8 HxW -> float32 NCHW in [0,1], normalized with the classic MNIST mean/std."""
    x = images.astype(np.float32) / 255.0
    x = (x - 0.1307) / 0.3081
    return x[:, None, :, :]


class DatasetBundle:
    """Everything an experiment needs: train/test tensors + metadata."""

    def __init__(self, train_n_per_class=1200, test_n_per_class=200, seed=0):
        tx, ty, vx, vy = load_mnist_raw()
        self.seed = seed
        tr_x, tr_y = stratified_subset(tx, ty, train_n_per_class, seed)
        te_x, te_y = stratified_subset(vx, vy, test_n_per_class, seed)
        # raw copies kept so attacks can manipulate pixels/labels precisely
        self.train_raw_x, self.train_raw_y = tr_x, tr_y.copy()
        self.test_raw_x, self.test_raw_y = te_x, te_y
        self.train_x = to_float(tr_x)
        self.train_y = tr_y.astype(np.int64)
        self.test_x = to_float(te_x)
        self.test_y = te_y.astype(np.int64)
        self.n_classes = 10

    def refloat(self):
        """Re-derive normalized tensors after raw-space poisoning."""
        self.train_x = to_float(self.train_raw_x)
        self.train_y = self.train_raw_y.astype(np.int64)


if __name__ == "__main__":
    b = DatasetBundle()
    print("train:", b.train_x.shape, b.train_y.shape)
    print("test :", b.test_x.shape, b.test_y.shape)
    print("class counts train:", np.bincount(b.train_y))
    print("class counts test :", np.bincount(b.test_y))
    print("pixel range:", float(b.train_x.min()), float(b.train_x.max()))