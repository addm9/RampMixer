# Datasets and preparation

## Sources

- **UNISOLAR:** [CDAClab dataset](https://www.kaggle.com/datasets/cdaclab/unisolar), described by Wimalaratne et al. (2022), DOI [10.1109/HSI55341.2022.9869474](https://doi.org/10.1109/HSI55341.2022.9869474).
- **HIT05667:** an individual-module series in the NREL collection [Data for Validating Models for PV Module Performance](https://www.osti.gov/dataexplorer/biblio/dataset/1811521).

Obtain measured data from the original providers and follow their citation and use conditions. This release includes the synthetic example only; measured data and trained artifacts are not redistributed.

## Expected inputs

The preparation script accepts the author's prepared CSV schemas below. It starts from timestamp-aligned CSV inputs; upstream site aggregation, weather alignment, and previous cleaning are not performed by this script. The [HIT05667 file in the PKINN repository](https://github.com/addm9/Photovoltaic_Knowledge_Informed_Neural_Network/blob/main/Dataset/Cocoa_HIT05667.csv) illustrates the expected module-data schema.

**UNISOLAR:** timestamp column `Timestamp`, power column `Power` (kW), and these 10 environmental columns:

```text
ghi, dni, cloud_opacity, air_temp, relative_humidity,
wind_speed_10m, wind_direction_10m, albedo, azimuth, precipitable_water
```

These names retain the supplied CSV meanings. In particular, `precipitable_water` is not atmospheric pressure.

**HIT05667:** timestamp column `date`, power column `Pmp (W)`, and these 7 environmental columns:

| Source column | Prepared column |
| --- | --- |
| `Global horizontal irradiance (W/m2)` | `ghi` |
| `Direct normal irradiance (W/m2)` | `dni` |
| `Diffuse horizontal irradiance (W/m2)` | `dhi` |
| `POA irradiance CMP22 pyranometer (W/m2)` | `poa_irradiance` |
| `Dry bulb temperature (degC)` | `air_temp` |
| `Relative humidity (%RH)` | `relative_humidity` |
| `Atmospheric pressure (mb)` | `atmospheric_pressure` |

The prepared input has columns `timestamp`, `power`, and the ordered drivers named in its configuration. Missing drivers are not synthesized. Time values retain the source clock convention without timezone conversion.

## Prepare a measured-data example

From the repository root:

```bash
python code/prepare_data.py --dataset unisolar --source /path/to/UNI.csv
python code/prepare_data.py --dataset hit05667 --source /path/to/Cocoa_HIT05667.csv
```

The script sorts timestamps, retains 08:00 inclusive to 17:00 exclusive, and selects the first 45 observed dates. It preserves the selected numeric values and saves source hashes, date coverage, and column mappings alongside the excerpt in `Dataset/local/`. Existing excerpts are not overwritten. This local directory is ignored by Git.

Invalid timestamps, duplicate records, and nonfinite values require explicit correction upstream. The demo does not interpolate or resample. Missing daytime observations and missing calendar days break forecasting windows.

## Synthetic example

`example/synthetic.csv` contains 720 simulated observations at 15-minute intervals, three illustrative environmental variables, and power in arbitrary units. Its deterministic generator is included in `code/prepare_data.py`. It provides a small executable test case and is not intended for evaluating forecasting performance on measured PV data.
