"""Independently recalculate demo metrics and replay a historical-only prediction."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from predict import predict
from rampmixer.pipeline import metrics


def verify(run_dir):
    root = Path(run_dir)
    rows = pd.read_csv(root/"predictions.csv")
    saved = json.loads((root/"metrics.json").read_text())["pooled"]
    recomputed = metrics(rows.actual.to_numpy(), rows.predicted.to_numpy())
    for key, value in saved.items():
        if value is None:
            if recomputed[key] is not None:
                raise AssertionError(f"Mismatch in {key}")
        else:
            np.testing.assert_allclose(value, recomputed[key], rtol=1e-8, atol=1e-8)
    with np.load(root/"frontend_example.npz", allow_pickle=False) as f:
        np.testing.assert_allclose(f["modes"].sum(-1), f["power"], rtol=1e-5, atol=1e-4)
        if not np.all((f["cam"] >= 0) & (f["cam"] <= 1)):
            raise AssertionError("CAM outside valid range")
    diagnostics = pd.read_csv(root/"crd_diagnostics.csv")
    if np.any(diagnostics.final_objective > diagnostics.initial_objective*(1+1e-8)+1e-8):
        raise AssertionError("CRD objective increased")
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in ("model.pt", "frontend.pkl")}
    with tempfile.TemporaryDirectory(prefix="rampmixer_verify_") as tmp:
        values = predict(root, root/"history_example.npz", Path(tmp)/"replay.npy")
    reference = rows[rows["sample"] == 0].sort_values("step").predicted.to_numpy()
    # Single-sample and minibatch matmul may differ by float32 round-off.
    np.testing.assert_allclose(values[0], reference, rtol=1e-4, atol=1e-4)
    for name, original in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != original:
            raise AssertionError(f"Inference changed fitted artifact: {name}")
    print("PASS: recomputed metrics, historical reconstruction, CRD objective, prediction replay, frozen artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    verify(parser.parse_args().run)
