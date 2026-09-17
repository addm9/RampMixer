"""Run a small end-to-end RampMixer experiment. No manuscript metrics are embedded."""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
import pickle
import platform
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import numba
import torch
from torch.utils.data import DataLoader, TensorDataset
from rampmixer.data import (PARTITIONS, validate_frame, chronological_split,
                           sample_indices, select_windows, check_target_disjoint)
from rampmixer.pipeline import fit_frontend, frontend, metrics
from rampmixer.ttm import RampMixer

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run(config_path, output=None, epochs=None, device_name="cpu"):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if epochs is not None:
        cfg["epochs"] = epochs
    if cfg["epochs"] < 1 or cfg["patience"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("Training counts must be positive")
    if cfg["length"] % cfg["patch"] or cfg["reference_power"] <= 0 or cfg["alpha0"] <= 0:
        raise ValueError("Invalid patch, reference power or bandwidth configuration")
    if len(cfg["max_samples"]) != 3:
        raise ValueError("Specify one sample budget for each partition")
    data_path = ROOT/cfg["data"]
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {cfg['data']}. See code/prepare_data.py and Dataset/README.md.")
    output = Path(output) if output else ROOT/"outputs"/(cfg["dataset"]+"_"+time.strftime("%Y%m%d_%H%M%S"))
    if output.exists():
        raise FileExistsError(f"Use a new output directory; existing results are preserved: {output}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    d = validate_frame(pd.read_csv(data_path), cfg["drivers"], cfg["minutes"])
    times = d.timestamp.to_numpy(dtype="datetime64[ns]")
    split = chronological_split(times, cfg["train_fraction"], cfg["validation_fraction"])
    all_origins = sample_indices(times, split, cfg["length"], cfg["horizon"], cfg["minutes"])
    origins = select_windows(all_origins, split, cfg["max_samples"])
    target_counts = check_target_disjoint(times, origins, split, cfg["horizon"])
    partition = split[origins]
    response, alpha, scale, fit_info = fit_frontend(d, split, cfg)
    p, e = d.power.to_numpy(float), d[cfg["drivers"]].to_numpy(float)
    power = np.stack([p[i-cfg["length"]:i] for i in origins])
    environment = np.stack([e[i-cfg["length"]:i] for i in origins])
    target = np.stack([p[i:i+cfg["horizon"]] for i in origins])
    output.mkdir(parents=True)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                   pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                   numba=numba.__version__, device=str(device),
                   gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    protocol = dict(scope="small fixed-configuration demo; not reproduction of manuscript tables",
                    config=cfg, runtime=runtime,
                    data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    training_only_frontend=fit_info, power_scale=scale,
                    forecast_targets_disjoint=True, unique_target_timestamps=target_counts,
                    partitions={k: dict(rows=int(np.sum(split == k)),
                                        selected_windows=int(np.sum(partition == k)),
                                        available_windows=int(np.sum(split[all_origins] == k)),
                                        first_timestamp=str(times[split == k][0]),
                                        last_timestamp=str(times[split == k][-1])) for k in PARTITIONS})
    write_json(output/"protocol.json", protocol)
    print(f"{cfg['dataset']}: fitting CAM on {fit_info['training_rows']} training rows", flush=True)
    print(f"Computing independent historical-window CRD for {len(origins)} samples...", flush=True)
    inputs, diagnostics, cam = frontend(power, environment, response, alpha, scale, cfg)
    columns = ["iterations", "relative_change", "center_change", "tolerance_met",
               "initial_objective", "final_objective", "max_accepted_increase", "stalled_updates"]
    diagnostic_table = pd.DataFrame(diagnostics, columns=columns)
    diagnostic_table.insert(0, "partition", partition)
    diagnostic_table.insert(1, "first_target", times[origins])
    diagnostic_table.to_csv(output/"crd_diagnostics.csv", index=False)
    unresolved = int(np.sum(diagnostics[:, 3] == 0))
    if unresolved:
        print(f"CRD: {unresolved} windows reached the iteration limit; see crd_diagnostics.csv.", flush=True)
    with (output/"frontend.pkl").open("wb") as handle:
        pickle.dump(dict(response=response, alpha=alpha, scale=scale), handle)
    tensors = [torch.from_numpy(a) for a in inputs]
    targets = torch.from_numpy((target/scale).astype(np.float32))
    loaders = {}
    for name in PARTITIONS:
        mask = partition == name
        ds = TensorDataset(*(t[mask] for t in tensors), targets[mask])
        loaders[name] = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=name == "train",
                                   generator=torch.Generator().manual_seed(cfg["seed"]), num_workers=0)
    model = RampMixer(len(cfg["drivers"])+1, length=cfg["length"], horizon=cfg["horizon"], patch=cfg["patch"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        train_sse, train_count = 0., 0
        for batch in loaders["train"]:
            m, desc, context, y = [v.to(device) for v in batch]
            optimizer.zero_grad()
            loss = (model(m, desc, context)-y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            train_sse += loss.item()*y.numel()
            train_count += y.numel()
        model.eval()
        val_sse, val_count = 0., 0
        with torch.no_grad():
            for batch in loaders["validation"]:
                m, desc, context, y = [v.to(device) for v in batch]
                val_sse += (model(m, desc, context)-y).square().sum().item()
                val_count += y.numel()
        val_rmse = float(np.sqrt(val_sse/val_count)*scale)
        if not np.isfinite(val_rmse):
            raise FloatingPointError("Nonfinite validation metric")
        history.append(dict(epoch=epoch, train_rmse=float(np.sqrt(train_sse/train_count)*scale),
                            validation_rmse=val_rmse))
        if val_rmse < best:
            best, stale = val_rmse, 0
            torch.save(model.state_dict(), output/"model.pt")
        else:
            stale += 1
        print(f"Epoch {epoch}: validation RMSE={val_rmse:.4f} {cfg['unit']}", flush=True)
        if stale >= cfg["patience"]:
            break
    pd.DataFrame(history).to_csv(output/"training_history.csv", index=False)
    model.load_state_dict(torch.load(output/"model.pt", map_location=device, weights_only=True))
    model.eval()
    prediction = []
    with torch.no_grad():
        for batch in loaders["test"]:
            prediction.append(model(*[v.to(device) for v in batch[:3]]).cpu().numpy()*scale)
    # Serialize and score the same float64 values, avoiding float32 CSV rounding drift.
    prediction = np.concatenate(prediction).astype(np.float64)
    truth = target[partition == "test"]
    test_origins = origins[partition == "test"]
    records = []
    for k, origin in enumerate(test_origins):
        for step in range(cfg["horizon"]):
            records.append(dict(sample=k, origin=str(times[origin-1]), step=step+1,
                                target_time=str(times[origin+step]), actual=truth[k, step], predicted=prediction[k, step]))
    pd.DataFrame(records).to_csv(output/"predictions.csv", index=False)
    result = dict(scope="demo test windows, pooled across all forecast steps", unit=cfg["unit"],
                  pooled=metrics(truth, prediction),
                  per_step={str(h+1): metrics(truth[:, h], prediction[:, h]) for h in range(cfg["horizon"])})
    write_json(output/"metrics.json", result)
    test_ix = np.flatnonzero(partition == "test")[0]
    np.savez_compressed(output/"history_example.npz", power=power[test_ix:test_ix+1],
                        environment=environment[test_ix:test_ix+1], drivers=np.asarray(cfg["drivers"]))
    np.savez_compressed(output/"frontend_example.npz", cam=cam[test_ix],
                        modes=inputs[0][test_ix]*scale, context=inputs[2][test_ix]*scale,
                        descriptors=inputs[1][test_ix], power=power[test_ix])
    write_json(output/"summary.json", dict(completed=True, epochs_run=len(history),
               parameters=sum(p.numel() for p in model.parameters()), best_validation_rmse=best,
               crd_windows_at_iteration_limit=unresolved, elapsed_seconds=time.perf_counter()-started,
               test=result["pooled"], unit=cfg["unit"]))
    print(f"Finished demo. Results: {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE/"configs/synthetic_demo.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    run(args.config, args.output, args.epochs, args.device)
