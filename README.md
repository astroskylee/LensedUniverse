# LensedUniverse

Cosmology forecasting utilities for strong-lensing data, including a combined
DSPL + lens-kinematics + lensed SNe pipeline.

## Contents
- `Combine_forecast.py`: main forecasting pipeline.
- `slcosmo/`: core models and tools (refactored).
- `Combine_forecast_test.ipynb`: smoke test with reduced settings.
- `environment_check.ipynb`: server-side environment report.
- `data/`: input catalogs and lookup tables.

## Quick start (test mode)
Use the lightweight test mode to validate the setup on a new machine:

```bash
export COMBINE_FORECAST_TEST=1
python Combine_forecast.py
```

This writes `combine_forecast_test_output*.nc/csv` in the working directory.

## Other forecast inputs
`Combine_forecast.py` expects LSST/Euclid files under `../SLCOSMO/other_forecast`
by default. Override with:

```bash
export OTHER_FORECAST_DIR=/path/to/other_forecast
```

## Requirements
See `requirements.txt` for a minimal set of Python dependencies.

## License
CC BY 4.0 (see `LICENSE`).
