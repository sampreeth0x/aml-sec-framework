"""Attack Module F — model artifact supply-chain integrity.

Lifecycle: train -> serialize -> registry (signed manifest) -> deploy -> verify.

The attacker tamper is a *real byte-level modification* of the serialized model
(one weight tensor perturbed slightly, mimicking a compromised registry or
stolen artifact). Integrity is enforced with SHA-256 content hashes plus an
HMAC-SHA256 signed manifest keyed by a deployment secret.

Everything here is real crypto on real model bytes.
"""
import hashlib
import hmac
import json
import os
import pickle
import struct
from pathlib import Path

import numpy as np

REGISTRY_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "registry"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize_model(model) -> bytes:
    """Deterministic serialization: sorted parameter names, float32 payload."""
    state = model.state_dict()
    buf = bytearray()
    for name in sorted(state.keys()):
        t = state[name].detach().cpu().contiguous()
        name_b = name.encode()
        buf += struct.pack(">I", len(name_b)) + name_b
        buf += struct.pack(">I", t.numel())
        buf += t.numpy().tobytes()
    return bytes(buf)


def sign_manifest(model_id: str, payload: bytes, key: bytes, metadata: dict) -> dict:
    """Build the registry manifest: content hash + HMAC signature over (id, hash, metadata)."""
    content_hash = _sha256(payload)
    canonical = json.dumps(
        {"model_id": model_id, "sha256": content_hash, "metadata": metadata},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    signature = hmac.new(key, canonical, hashlib.sha256).hexdigest()
    return {
        "model_id": model_id,
        "sha256": content_hash,
        "metadata": metadata,
        "hmac_sha256": signature,
    }


def publish_to_registry(model, model_id: str, metadata: dict, key: bytes) -> Path:
    """Store payload + signed manifest. This stands in for a real model registry."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    payload = serialize_model(model)
    manifest = sign_manifest(model_id, payload, key, metadata)
    path = REGISTRY_DIR / f"{model_id}.bin"
    path.write_bytes(payload)
    (REGISTRY_DIR / f"{model_id}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


def tamper_payload(path: Path, rng_seed=7) -> bytes:
    """Modify real bytes: perturb 4 float32 values deep inside the first tensor block.
    Returns (tampered_bytes, n_bytes_changed, description)."""
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    payload = bytearray(path.read_bytes())
    # pick an offset well past the headers (first ~1KB in) and flip a float's mantissa bits
    offset = int(rng.integers(1024, min(4096, len(payload) - 4)))
    val = struct.unpack_from("<f", payload, offset)[0]
    val_tampered = float(np.nextafter(np.float32(val), np.float32(np.inf)))  # 1-ulp change
    struct.pack_into("<f", payload, offset, val_tampered)
    return bytes(payload), 4, f"1-ulp float modification at byte offset {offset} ({val!r} -> {val_tampered!r})"


def verify_deployment(payload: bytes, manifest: dict, key: bytes) -> dict:
    """The gate the deployment stage must pass. Recomputes hash + HMAC, compares."""
    actual_hash = _sha256(payload)
    hash_ok = hmac.compare_digest(actual_hash, manifest["sha256"])
    canonical = json.dumps(
        {"model_id": manifest["model_id"], "sha256": actual_hash, "metadata": manifest["metadata"]},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    sig_ok = hmac.compare_digest(
        hmac.new(key, canonical, hashlib.sha256).hexdigest(), manifest["hmac_sha256"]
    )
    return {"content_hash_ok": bool(hash_ok), "signature_ok": bool(sig_ok),
            "verdict": "PASS" if (hash_ok and sig_ok) else "TAMPER_DETECTED"}


def load_tampered_state(payload: bytes, n_classes=10):
    """Deserialize (possibly tampered) payload back into a live model."""
    from amlsec.models import make_model
    offset = 0
    state = {}
    while offset < len(payload):
        (nlen,) = struct.unpack_from(">I", payload, offset)
        offset += 4
        name = payload[offset:offset + nlen].decode()
        offset += nlen
        (nel,) = struct.unpack_from(">I", payload, offset)
        offset += 4
        arr = np.frombuffer(payload[offset:offset + nel * 4], dtype=np.float32).copy()
        offset += nel * 4
        state[name] = arr
    # reshape using clean reference shapes
    ref = make_model("cnn", n_classes).state_dict()
    sd = {}
    for k, v in ref.items():
        sd[k] = torch_from(state.get(k, v)).reshape(v.shape)
    model = make_model("cnn", n_classes)
    model.load_state_dict(sd)
    model.eval()
    return model


def torch_from(arr):
    import torch
    return torch.from_numpy(arr)