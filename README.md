# RampMixer

This repository provides a PyTorch reference implementation and runnable examples of RampMixer for photovoltaic (PV) power forecasting under ramp conditions. The framework uses environmental attribution to guide power decomposition and learns the temporal dynamics of the resulting contribution-aligned modes.

The supplied configurations are small-data demonstrations of the workflow, not a reproduction package for the manuscript's benchmark tables.

## Repository structure

```text
Dataset/
  example/          Synthetic example for running the demo
  README.md         Dataset sources, input fields, and preparation
code/
  rampmixer/        CAM, CRD, and the ramp-oriented TTM
  configs/          Synthetic, UNISOLAR, and HIT05667 demo configurations
  tests/            Numerical and data-protocol tests
  train.py          Training and evaluation
  predict.py        Prediction from saved models
  prepare_data.py   Preparation of measured-data excerpts
  verify_run.py     Independent verification of saved results
  requirements.txt Runtime dependencies
README.md
```

## Dataset overview

The measured-data examples use historical PV power and environmental observations from two sources:

| Dataset | Forecasting scale | Sampling interval | Power unit |
| --- | --- | --- | --- |
| [UNISOLAR](https://www.kaggle.com/datasets/cdaclab/unisolar) | Aggregated PV systems | 15 min | kW |
| [HIT05667](https://www.osti.gov/dataexplorer/biblio/dataset/1811521) | Individual PV module | 5 min | W |

Measured datasets are obtained separately from their providers. The included `Dataset/example/synthetic.csv` contains **simulated observations** for testing the workflow; it is not a subset of either measured dataset. The measured-data configurations use the existing environmental fields listed in [Dataset/README.md](Dataset/README.md).

## About RampMixer

- **Coupling Attribution Matrix (CAM):** represents time-varying driver contributions using main effects and selected pairwise interactions in a fitted environmental response model.
- **Contribution-Guided Ramp Decomposition (CRD):** uses CAM-derived contribution targets to constrain coupled variational optimization and construct contribution-aligned power modes.
- **Ramp-oriented Tiny Time Mixer (TTM):** encodes each mode's temporal dynamics and retrieves relevant historical power, trend, and fluctuation information for forecasting.

The response model, normalization, and bandwidth coefficients are fitted using training observations. At prediction time, CAM and CRD are evaluated independently within each historical input window, while the fitted frontend settings and network weights remain fixed. Numerical choices are documented in [code/IMPLEMENTATION.md](code/IMPLEMENTATION.md).

## Quick start

Run the following commands from the repository root in an isolated Python environment. The example has been tested on CPU with Python 3.9.19 and PyTorch 2.2.2. A GPU is not required.

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r code/requirements.txt
python code/run_tests.py
python code/train.py --output outputs/example
python code/verify_run.py --run outputs/example
```

The default configuration uses 96 historical observations, a 12-step forecast, and up to eight training epochs. Results are saved in a new output directory. The first run may take longer while the CRD solver is compiled.

To run a measured-data example after preparing the corresponding input file:

```bash
python code/train.py --config code/configs/unisolar_demo.json --output outputs/unisolar
python code/train.py --config code/configs/hit05667_demo.json --output outputs/hit05667
```

## Prediction and outputs

```bash
python code/predict.py --run outputs/example --history outputs/example/history_example.npz --output outputs/example/replay.npy
```

Each run saves the selected model, fitted frontend, configuration, training history, per-step predictions, evaluation metrics, and CRD stopping diagnostics. `verify_run.py` recalculates metrics and checks prediction replay without retraining. Predictions are expressed in the input dataset's power unit. Only load checkpoints from trusted sources.
