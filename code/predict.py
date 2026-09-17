"""Predict from saved demo artifacts and historical observations only.

Only load artifacts that you trust: frontend.pkl uses Python pickle.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from rampmixer.pipeline import frontend
from rampmixer.ttm import RampMixer


def predict(run_dir, history, output):
    run_dir, history, output = Path(run_dir), Path(history), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if output.suffix != ".npy":
        raise ValueError("Use an output filename ending in .npy")
    protocol = json.loads((run_dir/"protocol.json").read_text(encoding="utf-8"))
    cfg = protocol["config"]
    with (run_dir/"frontend.pkl").open("rb") as handle:
        fitted = pickle.load(handle)
    with np.load(history, allow_pickle=False) as batch:
        if batch["drivers"].tolist() != cfg["drivers"]:
            raise ValueError("Environmental driver names/order do not match the training configuration")
        inputs, diagnostics, _ = frontend(batch["power"], batch["environment"], cfg=cfg, **fitted)
    model = RampMixer(len(cfg["drivers"])+1, length=cfg["length"], horizon=cfg["horizon"], patch=cfg["patch"])
    model.load_state_dict(torch.load(run_dir/"model.pt", map_location="cpu", weights_only=True))
    model.eval()
    torch.set_num_threads(4)
    with torch.no_grad():
        values = model(*[torch.from_numpy(a) for a in inputs]).numpy()*fitted["scale"]
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, values, allow_pickle=False)
    print(f"Saved {values.shape} predictions in {cfg['unit']}; no retraining performed.")
    if np.any(diagnostics[:, 3] == 0):
        print("Some inference-window CRD solves reached the configured iteration limit.")
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict(args.run, args.history, args.output)
