"""Train-only environmental fitting and independent historical-window frontend."""
import numpy as np
from .cam import ANOVAResponse, lag_states, causal_trend
from .crd import decompose
from .data import contiguous_blocks


def fit_frontend(frame, split, cfg):
    train = frame.loc[split == "train"]
    p = train.power.to_numpy(float)
    e = train[cfg["drivers"]].to_numpy(float)
    if len(p) < 3 or np.std(p) < 1e-8:
        raise ValueError("Training power must contain enough nonconstant observations")
    states = np.empty((len(p), len(cfg["drivers"]), cfg["lag"]+1))
    fluctuation = np.empty(len(p))
    ramp = np.zeros(len(p), dtype=bool)
    for ix in contiguous_blocks(train.timestamp.to_numpy(), cfg["minutes"]):
        states[ix] = lag_states(e[ix], cfg["lag"])
        fluctuation[ix] = p[ix]-causal_trend(p[ix], cfg["sg_width"])
        ramp[ix[1:]] = np.abs(np.diff(p[ix]))/(cfg["reference_power"]*cfg["minutes"]/60) >= .4
    response = ANOVAResponse(gamma=cfg["gamma"], rank=cfg["basis_rank"], ridge=cfg["ridge"])
    response.fit(states, fluctuation)
    cam = response.attribute(states)
    average = cam[ramp].mean(0) if ramp.any() else cam.mean(0)
    alpha = cfg["alpha0"]*(1+cfg["beta"]*average)
    scale = float(p.std())
    return response, alpha, scale, dict(
        training_rows=len(p), training_ramp_observations=int(ramp.sum()),
        no_ramp_fallback=not bool(ramp.any()),
        alpha=alpha.tolist(), selected_pairs=[list(pair) for pair in response.pairs],
        response_training_r2=response.train_r2, power_scale_training_std=scale)


def frontend(power, environment, response, alpha, scale, cfg):
    """Only supplied histories are visible. Fitted objects are not updated."""
    power = np.asarray(power, dtype=float)
    environment = np.asarray(environment, dtype=float)
    expected = (len(power), cfg["length"], len(cfg["drivers"]))
    if power.ndim != 2 or power.shape[1] != cfg["length"] or environment.shape != expected:
        raise ValueError("Historical input shapes do not match the run configuration")
    if not np.isfinite(power).all() or not np.isfinite(environment).all():
        raise ValueError("Historical observations must be finite")
    weights = np.stack([response.attribute(lag_states(window, cfg["lag"])) for window in environment])
    p = power/scale
    modes, centers, diagnostics = decompose(
        p, weights, alpha, overlap=cfg["overlap"], max_iter=cfg["crd_max_iter"], tol=cfg["crd_tol"])
    trend = np.stack([causal_trend(window, cfg["sg_width"]) for window in p])
    context = np.stack([p, trend, p-trend], axis=-1)
    descriptors = np.zeros((len(power), len(alpha)+1, 2))
    descriptors[:, :-1, 0] = alpha/cfg["alpha0"]
    descriptors[:, :-1, 1] = centers/np.pi
    arrays = (modes, descriptors, context)
    if not all(np.isfinite(a).all() for a in arrays):
        raise FloatingPointError("Nonfinite decomposition outputs")
    return tuple(a.astype(np.float32) for a in arrays), diagnostics, weights


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.shape != prediction.shape or not truth.size:
        raise ValueError("Metric arrays must have identical, nonempty shapes")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric arrays must be finite")
    error = prediction-truth
    denominator = np.sum((truth-truth.mean())**2)
    return dict(RMSE=float(np.sqrt(np.mean(error**2))), MAE=float(np.mean(abs(error))),
                R2=float(1-np.sum(error**2)/denominator) if denominator > 0 else None,
                MAPE_percent=float(100*np.mean(abs(error)/np.maximum(abs(truth), 1e-6))))
