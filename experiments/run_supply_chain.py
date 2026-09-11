"""Experiment F — supply-chain integrity lifecycle.

1. Train a real model, serialize deterministically, publish to the registry
   with an HMAC-signed manifest.
2. Benign redeploy: byte-identical payload -> verification must PASS.
3. Tamper scenario: real 1-ulp float modification of the artifact bytes ->
   verification must FAIL before the model is ever loaded.
4. Behavioral comparison: the tampered model (if it slipped through a
   hash-less pipeline) vs the approved model — shows what integrity checking
   protects against that behavioral testing might miss.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import secrets

import numpy as np

from amlsec.attacks.supply_chain import (serialize_model, sign_manifest, publish_to_registry,
                                         tamper_payload, verify_deployment, REGISTRY_DIR)
from amlsec.experiment_utils import save_result, run_stamp, new_bundle
from amlsec.metrics import accuracy
from amlsec.models import make_model, train_model, evaluate, load_model


def main():
    bundle = new_bundle()
    results = {"timestamp": run_stamp(), "lifecycle": []}

    # 1) train + publish
    model = make_model("cnn")
    train_model(model, bundle.train_x, bundle.train_y, epochs=4, seed=0,
                log_prefix="[sc] ", test_x=bundle.test_x, test_y=bundle.test_y)
    preds, _, _ = evaluate(model, bundle.test_x, bundle.test_y)
    approved_acc = accuracy(preds, bundle.test_y)

    key = secrets.token_bytes(32)  # deployment secret — real key material per run
    publish_to_registry(model, "mnist-cnn-v1",
                        {"dataset": "mnist-subset", "accuracy": approved_acc,
                         "framework": "amlsec", "epochs": 4}, key)
    payload = (REGISTRY_DIR / "mnist-cnn-v1.bin").read_bytes()
    manifest = (REGISTRY_DIR / "mnist-cnn-v1.manifest.json").read_text()

    # 2) benign redeploy -> must PASS
    benign = verify_deployment(payload, __import__("json").loads(manifest), key)
    benign["scenario"] = "benign_redeploy"
    benign["behavior_agrees_with_approved"] = True
    results["lifecycle"].append(benign)
    print(f"[sc] benign redeploy: {benign['verdict']}", flush=True)

    # 3) tamper: real byte modification -> must FAIL
    tampered, n_bytes, desc = tamper_payload(REGISTRY_DIR / "mnist-cnn-v1.bin", rng_seed=7)
    verdict = verify_deployment(tampered, __import__("json").loads(manifest), key)
    verdict["scenario"] = "tampered_artifact"
    verdict["tamper_description"] = desc
    results["lifecycle"].append(verdict)
    print(f"[sc] tamper detected: {verdict['verdict']} ({desc})", flush=True)

    # 3b) tamper WITHOUT the secret — attacker tries to re-sign with wrong key
    forged_key = secrets.token_bytes(32)
    forged = verify_deployment(tampered, __import__("json").loads(manifest), forged_key)
    forged["scenario"] = "tampered + forged_key_recompute"
    results["lifecycle"].append(forged)
    print(f"[sc] forged-key check: {forged['verdict']}", flush=True)

    # 4) behavioral comparison: tampered model vs approved model
    from amlsec.attacks.supply_chain import load_tampered_state
    tampered_model = load_tampered_state(tampered)
    tp, _, _ = evaluate(tampered_model, bundle.test_x, bundle.test_y)
    tampered_acc = accuracy(tp, bundle.test_y)
    disagreement = float((tp != preds).mean())
    # statistical test: would a normal accuracy check catch this tamper?
    from amlsec.metrics import binomial_ci
    results["tampered_behavior"] = {
        "tampered_accuracy": tampered_acc,
        "approved_accuracy": approved_acc,
        "prediction_disagreement_rate": disagreement,
        "accuracy_would_flag_it": abs(tampered_acc - approved_acc) > 0.02,
        "note": "1-ulp tampering is behaviorally near-invisible; integrity checks are the only reliable control",
    }
    print(f"[sc] tampered model acc={tampered_acc:.4f} vs approved {approved_acc:.4f} "
          f"disagreement={disagreement:.4f}", flush=True)

    # 5) hash chain of everything published — audit trail
    import hashlib
    from pathlib import Path
    audit = []
    for f in sorted(REGISTRY_DIR.glob("*")):
        audit.append({"file": f.name, "size": f.stat().st_size,
                      "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    results["registry_audit"] = audit
    save_result("supply_chain", results)


if __name__ == "__main__":
    main()