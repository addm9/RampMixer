"""Prepare local, auditable excerpts without modifying the source CSV files."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rampmixer.data import validate_frame

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "unisolar": {
        "time": "Timestamp", "power": "Power", "minutes": 15, "unit": "kW",
        "source_url": "https://www.kaggle.com/datasets/cdaclab/unisolar",
        "drivers": {k: k for k in ["ghi", "dni", "cloud_opacity", "air_temp", "relative_humidity",
                                   "wind_speed_10m", "wind_direction_10m", "albedo", "azimuth", "precipitable_water"]},
    },
    "hit05667": {
        "time": "date", "power": "Pmp (W)", "minutes": 5, "unit": "W",
        "source_url": "https://www.osti.gov/dataexplorer/biblio/dataset/1811521",
        "drivers": {
            "Global horizontal irradiance (W/m2)": "ghi",
            "Direct normal irradiance (W/m2)": "dni",
            "Diffuse horizontal irradiance (W/m2)": "dhi",
            "POA irradiance CMP22 pyranometer (W/m2)": "poa_irradiance",
            "Dry bulb temperature (degC)": "air_temp",
            "Relative humidity (%RH)": "relative_humidity",
            "Atmospheric pressure (mb)": "atmospheric_pressure",
        },
    },
}


def prepare(dataset, source, days=45):
    spec = DATASETS[dataset]
    path = ROOT/"Dataset"/"local"/f"{dataset}.csv"
    manifest = path.with_suffix(".provenance.json")
    if path.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite existing excerpt: {path}")
    if days < 10:
        raise ValueError("At least ten observed dates are required")
    d = pd.read_csv(source, usecols=[spec["time"], spec["power"]]+list(spec["drivers"]))
    input_rows = len(d)
    mapping = {spec["time"]: "timestamp", spec["power"]: "power", **spec["drivers"]}
    d = d.rename(columns=mapping)
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="raise")
    if d["timestamp"].isna().any() or d["timestamp"].duplicated().any():
        raise ValueError("Invalid or duplicate source timestamps; resolve upstream explicitly")
    d = d.sort_values("timestamp")
    t = d["timestamp"]
    d = d[(t.dt.hour >= 8) & (t.dt.hour < 17)].copy()
    selected_days = d["timestamp"].dt.normalize().unique()[:days]
    d = d[d["timestamp"].dt.normalize().isin(selected_days)]
    d = validate_frame(d, list(spec["drivers"].values()), spec["minutes"])
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    info = dict(dataset=dataset, input_filename=Path(source).name,
                source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                excerpt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                original_dataset_url=spec["source_url"], redistribution_status="local_only_pending_review",
                provenance="Excerpt of the author's supplied CSV; upstream merging/preprocessing not reconstructed here.",
                input_rows=input_rows, excerpt_rows=len(d), observed_dates=len(selected_days),
                first_timestamp=str(d.timestamp.min()), last_timestamp=str(d.timestamp.max()),
                unit=spec["unit"], column_mapping=mapping,
                operations=["parse and sort timestamps", "retain 08:00 <= time < 17:00",
                            f"first {days} observed dates", "select and rename columns without changing values"],
                missing_policy="No interpolation or resampling. Missing intervals break model windows.")
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {dataset}: {len(d)} rows, {len(selected_days)} dates -> {path.relative_to(ROOT)}")
    return path


def synthetic():
    """A deterministic numerical fixture, not measured PV data or a physical simulator."""
    path = ROOT/"Dataset"/"example"/"synthetic.csv"
    if path.exists():
        raise FileExistsError(f"Example already exists: {path}")
    rng = np.random.default_rng(20260917)
    frames = []
    for day in range(20):
        t = np.arange(36)
        cloud = np.clip(.35+.25*np.sin(t/3+day*.7)+.1*rng.normal(size=36), 0, 1)
        irradiance = np.maximum(0, 850*np.sin((t+3)*np.pi/44))*(1-.65*cloud)
        temperature = 21+5*np.sin(t/18)+day*.05+rng.normal(0, .25, 36)
        power = np.maximum(0, .12*irradiance*(1-.004*(temperature-25))
                           + 3*np.sin(t/2+day)+rng.normal(0, .5, 36))
        timestamp = pd.Timestamp("2020-01-01 08:00")+pd.Timedelta(days=day)+pd.to_timedelta(t*15, unit="m")
        frames.append(pd.DataFrame(dict(timestamp=timestamp, power=power, irradiance=irradiance,
                                        temperature=temperature, cloud_fraction=cloud)))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, float_format="%.8f")
    print(f"Created synthetic fixture: {path.relative_to(ROOT)}; NOT UNISOLAR/HIT05667")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "synthetic"], required=True)
    parser.add_argument("--source", type=Path, help="Author-prepared CSV in the documented source schema")
    parser.add_argument("--days", type=int, default=45)
    args = parser.parse_args()
    if args.dataset == "synthetic":
        synthetic()
    elif args.source is None:
        parser.error("--source is required for measured datasets")
    else:
        prepare(args.dataset, args.source, args.days)
