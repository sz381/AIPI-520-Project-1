# AIPI 520 – Project 1: Modeling Weather

Forecast of the hourly temperature at Raleigh-Durham International Airport (RDU)
for September 17-30, 2026, made only from data available before September 17,
12am Eastern: NOAA GHCNh observations since 2015 and archived ECMWF forecast
runs. The final model is a linear regression; a linear GAM is the second model.

## Project structure

```
.
├── data/
│   ├── raw/                    # Downloads (notebook 01): NOAA observations, ECMWF and GFS forecast runs
│   ├── interim/                # Cleaned hourly temperature (notebook 02)
│   ├── processed/              # Model-ready sample tables (notebooks 03, 04)
│   └── external/               # Sep 17-30 observations (notebook 05)
├── notebooks/                  # Numbered in run order
│   ├── 01_data_sourcing_new.ipynb                    # Downloads the observations and forecast runs
│   ├── 02_cleaning_eda_new.ipynb                     # Cleaning, quality checks, exploration
│   ├── 03_feature_engineering_new.ipynb              # Builds the sample table
│   ├── 04_linear_regression_modeling_new_new.ipynb   # Linear regressions, validation, final forecast
│   ├── 05_linear_regression_test_evaluation.ipynb    # Downloads Sep 17-30 and scores the forecasts
│   ├── 06_linear_gam.ipynb                           # Figures for the linear GAM
│   └── deprecated/             # Earlier versions, kept for reference
├── scripts/
│   ├── linear_gam.py                   # Fits and validates the linear GAM, freezes its forecast
│   ├── linear_gam_test_evaluation.py   # Scores the frozen GAM forecast on Sep 17-30
│   ├── results/                # Outputs of the scripts, and the frozen forecasts' hashes
│   └── deprecated/             # Earlier versions, kept for reference; they do not run from there
├── reports/                    # Forecasts, validation summary, figures/
├── docs/
│   ├── linear_gam.md           # The linear GAM: approach, inputs, how to run
│   └── data_checks.md          # Earlier checks on the inputs: publication times, the station break (cited in notebook 02)
├── models/                     # Fitted models (not tracked)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate.ps1
pip install -r requirements.txt
```

## Running it

The data and results these steps produce are already in the repository, so any
step can be run on its own. The one exception is notebook 06, which needs the
model file that `linear_gam.py` writes to `models/`.

```bash
# Notebooks, in order (01 and 05 download from NOAA and Open-Meteo)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_sourcing_new.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_cleaning_eda_new.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_feature_engineering_new.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_linear_regression_modeling_new_new.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_linear_regression_test_evaluation.ipynb

# Linear GAM (a few minutes), its test scores, then its figures
python scripts/linear_gam.py
python scripts/linear_gam_test_evaluation.py
jupyter nbconvert --to notebook --execute --inplace notebooks/06_linear_gam.ipynb
```

## Data

- **Observations**: NOAA NCEI Global Historical Climatology Network hourly
  (GHCNh), station `USW00013722`, routine reports at minute 51. A report at
  hh:51 is labelled hour hh.
- **Forecasts**: ECMWF IFS runs from the Open-Meteo Single Runs API, which
  returns each run as it was issued (weather data by Open-Meteo.com, CC BY 4.0).
  GFS runs are downloaded for comparison only.
- **Cutoff**: nothing observed or published at or after September 17, 12am
  Eastern enters a forecast. `docs/data_checks.md` records when each run was
  published.

## Frozen forecasts and the test window

| Forecast | File | Recorded hash |
|---|---|---|
| Linear regression | `scripts/results/final_forecast_M14.csv` | `scripts/results/final_forecast_M14.sha256` |
| Linear GAM | `reports/forecast_2026-09-17_to_2026-09-30_linear_gam.csv` | `scripts/results/final_forecast_linear_gam.sha256` |

Notebooks 04-06 and `linear_gam_test_evaluation.py` verify each forecast against
its recorded SHA-256 before using it. `.gitattributes` keeps CSV files
byte-identical across operating systems so that these checks also pass on
Windows.

The observations of September 17-30 are downloaded by notebook 05 into
`data/external/rdu_obs_2026-09-17_to_2026-09-30.csv`. That notebook is the only
code that writes the file. On October 5 NOAA had published 322 of the 336 hours;
re-run notebook 05 to pick up the rest.

## Conventions

- Never edit files in `data/raw/`; write derived data to `data/interim/` or
  `data/processed/`.
- Name notebooks `NN_short_description.ipynb` and keep them in run order.
- Split by forecast run and by time, never by row at random.
