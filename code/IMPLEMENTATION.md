# Implementation and experiment settings

## Release scope

This PyTorch demo implements the CAM–CRD–TTM workflow with small fixed-budget examples. It is not the original code used to generate the manuscript's benchmark results. It is a separate implementation from the TensorFlow environment described in the manuscript; framework, dataset fields, and example settings below refer to this release.

## Environmental attribution

The response model uses current observations and three lags per environmental variable. The target is power minus a one-sided quadratic Savitzky–Golay trend. Training filters and lag construction reset at overnight and missing-data boundaries.

Main terms use centered quadratic polynomial bases compressed by SVD to at most four dimensions per driver. Pairwise terms use tensor products of those bases. Pairs are selected using training-set absolute Pearson correlation greater than 0.4, and response coefficients are fitted jointly with ridge regularization of 100. These numerical choices specify the response estimator used here.

CAM assigns main-effect magnitudes to their variables and allocates pairwise interaction magnitudes according to the relative main-effect magnitudes of the two variables. The strengths are normalized with epsilon `1e-6`. Attributions are model-based, defined under a product-of-empirical-marginals reference measure; they are not identified physical causal contributions.

## Historical-window decomposition

For driver `i`, the contribution target is `CAM[t, i] * historical_power[t]`. The CRD objective contains reconstruction, contribution-target fidelity, spectral bandwidth, and absolute inner-product overlap terms. Modes are updated alternately in the real-FFT domain, with backtracking and spectral-centroid updates of center frequencies. Reflection padding uses historical values only.

Bandwidths follow `alpha_i = 2000 * (1 + 0.8 * mean_ramp_CAM_i)`. Calibration uses training observations with an absolute normalized power-change rate of at least 0.4 per hour. If no such observation is available, the training-wide mean CAM is used and the fallback is recorded. Bandwidths remain fixed during validation and testing.

The solver uses overlap coefficient 0.1, at most 500 iterations, and tolerance `1e-6` for relative mode change and center-frequency change. Stopping diagnostics are saved per window; meeting the stopping criterion is not proof of a global optimum. Driver modes and the residual reconstruct the historical power sequence.

## Temporal prediction

The network uses six-observation patches, hidden width 64, two temporal-mixing blocks, four attention heads, and a 128-unit prediction head. Bandwidth, center frequency, and learned driver identity form the mode descriptors. Temporal encoding shares parameters across modes without mixing their activations. Encoded modes query historical power, trend, and fluctuation tokens; component forecasts are summed.

The model is trained from scratch, not loaded from an IBM TTM checkpoint. It uses `M` driver modes and one residual; trend is a context input rather than an additional decomposed mode. Only the aggregated forecast is supervised, with no decomposition of future targets.

## Example protocol

| Setting | Value |
| --- | --- |
| History / horizon | 96 / 12 observations |
| Split | Whole observed dates, chronological 60% / 20% / 20% |
| Selected windows | Up to 64 training, 16 validation, and 16 test windows |
| Selection rule | Evenly spaced valid windows within each partition |
| Optimizer | Adam, learning rate 0.001, batch size 32 |
| Training | Up to 8 epochs, patience 4, seed 42 |
| Model selection | Lowest validation RMSE |

The response model, basis transformations, pair selection, power normalization, and bandwidths use training observations only. Validation and test histories can include already observed earlier partitions, but forecast target timestamps do not overlap across partitions.

Histories can bridge adjacent complete daytime blocks. Forecast targets cannot cross nights or split boundaries. Thus, 96 retained observations do not always represent 96 equal wall-clock intervals; spectral frequency is per retained observation. The examples use actual available drivers: 10 for UNISOLAR and 7 for HIT05667.

Saved RMSE and MAE use the dataset power unit. R² is a ratio; MAPE is a percentage with denominator `max(abs(actual), 1e-6)`. Pooled metrics count sample–horizon pairs. The example runs do not include hyperparameter search, repeated-seed statistics, or baseline rankings.

## Verification

Local checks use Python 3.9.19, PyTorch 2.2.2, and the dependencies pinned in `requirements.txt`. Run `python code/run_tests.py` from the repository root. Tests check effect allocation, historical boundaries, train-only fitting, independent mode encoding, CRD reconstruction and objective decrease, target separation, and numerical input validation. `verify_run.py` independently recalculates saved metrics and replays predictions with frozen artifacts. A configured Python 3.10 CI job requires execution on GitHub before its status can be confirmed.
