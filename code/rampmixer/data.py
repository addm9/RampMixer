"""Input validation and chronological, target-disjoint demo splits."""
import numpy as np
import pandas as pd

PARTITIONS = ("train", "validation", "test")


def validate_frame(frame, drivers, minutes):
    required = ["timestamp", "power"] + list(drivers)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not drivers or len(set(drivers)) != len(drivers):
        raise ValueError("Provide distinct environmental driver columns")
    if minutes <= 0 or 60 % minutes:
        raise ValueError("Sampling minutes must be a positive divisor of 60")
    d = frame.loc[:, required].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="raise")
    if d["timestamp"].isna().any() or d["timestamp"].duplicated().any():
        raise ValueError("Missing or duplicate timestamps must be resolved explicitly")
    d = d.sort_values("timestamp").reset_index(drop=True)
    if d.empty:
        raise ValueError("No observations")
    values = d[["power"] + list(drivers)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite observations: this demo does not interpolate missing data")
    t = d["timestamp"]
    if not ((t.dt.hour >= 8) & (t.dt.hour < 17)).all():
        raise ValueError("Expected daytime observations: 08:00 <= time < 17:00")
    if ((t.dt.minute % minutes != 0) | (t.dt.second != 0) | (t.dt.microsecond != 0)).any():
        raise ValueError("Timestamps are not aligned to the configured sampling grid")
    return d


def chronological_split(times, train_fraction=.6, validation_fraction=.2):
    """Split whole observed dates, before fitting any data-dependent transform."""
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Invalid chronological split fractions")
    days = np.asarray(times, dtype="datetime64[ns]").astype("datetime64[D]")
    unique = np.unique(days)
    a = int(len(unique)*train_fraction)
    b = int(len(unique)*(train_fraction+validation_fraction))
    if not 0 < a < b < len(unique):
        raise ValueError("Too few distinct dates for three chronological partitions")
    return np.where(days < unique[a], "train", np.where(days < unique[b], "validation", "test"))


def contiguous_blocks(times, minutes):
    t = np.asarray(times, dtype="datetime64[ns]")
    cuts = np.flatnonzero(np.diff(t) != np.timedelta64(minutes, "m")) + 1
    return np.split(np.arange(len(t)), cuts)


def sample_indices(times, split, length, horizon, minutes):
    """Historical windows may bridge COMPLETE adjacent retained daytime blocks.

    Targets stay within one day and one partition. Missing intraday observations
    and missing calendar days break windows. Overnight bridging is allowed only
    from the last configured daytime observation to 08:00 on the next day.
    """
    if length < 2 or horizon < 1:
        raise ValueError("Invalid historical length or horizon")
    t = np.asarray(times, dtype="datetime64[ns]")
    days = t.astype("datetime64[D]")
    step = np.timedelta64(minutes, "m")
    normal = np.diff(t) == step
    overnight = ((days[1:]-days[:-1] == np.timedelta64(1, "D"))
                 & (t[:-1] == days[:-1]+np.timedelta64(17*60-minutes, "m"))
                 & (t[1:] == days[1:]+np.timedelta64(8*60, "m")))
    bad = np.r_[0, np.cumsum(~(normal | overnight))]
    origins = []
    for i in range(length, len(t)-horizon+1):
        if np.any(split[i:i+horizon] != split[i]):
            continue
        if days[i-1] != days[i] or days[i] != days[i+horizon-1]:
            continue
        if bad[i+horizon-1] != bad[i-length]:
            continue
        origins.append(i)
    return np.asarray(origins, dtype=int)


def select_windows(origins, split, limits):
    """Evenly spaced indices per partition, never selected by error or ramp size."""
    chosen = []
    for name, limit in zip(PARTITIONS, limits):
        ix = origins[split[origins] == name]
        if not len(ix):
            raise ValueError(f"No valid {name} windows. Use more dates or inspect data gaps.")
        if limit < 1:
            raise ValueError("Demo sample limits must be positive")
        chosen.append(ix[np.linspace(0, len(ix)-1, min(len(ix), limit), dtype=int)])
    return np.concatenate(chosen)


def check_target_disjoint(times, origins, split, horizon):
    targets = {k: set() for k in PARTITIONS}
    for i in origins:
        if np.any(split[i:i+horizon] != split[i]):
            raise ValueError("Forecast targets cross a partition boundary")
        targets[split[i]].update(times[i:i+horizon])
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if targets[a] & targets[b]:
            raise ValueError(f"Target overlap between {a} and {b}")
    return {k: len(v) for k, v in targets.items()}
