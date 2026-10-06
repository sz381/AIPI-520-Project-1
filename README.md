# AIPI 520 – Project 1: Modeling Weather

Forecasting the hourly temperature at Raleigh-Durham International Airport (RDU)
for Sep 17-30, 2026, using only data from before Sep 17 12am Eastern: NOAA GHCNh
hourly observations (2015-01-01 through 2026-09-16) and archived ECMWF and GFS
forecast runs.

## Project structure

```
.
├── data/
│   ├── raw/ghcnh_rdu/<year>/   # Immutable downloads, one CSV per (Eastern) year
│   ├── raw/openmeteo_single_runs/  # Archived ECMWF and GFS 12z forecast runs
│   ├── raw/ghcnh_rdu_holdout/  # Observations of Sep 17-30, for final scoring only
│   ├── external/               # Reference data for diagnostics (ERA5), never a model input
│   ├── processed/              # Final, model-ready datasets
├── notebooks/                  # Exploratory work, numbered in run order
│   ├── 01_data_sourcing.ipynb
│   └── 06_linear_gam.ipynb     # Plots and comparisons of the linear GAM's results
├── scripts/                    # Runnable, reproducible entry points
│   ├── download_data.py        # Downloads raw GHCNh data for RDU
│   ├── download_nwp.py         # Downloads the forecast runs from Open-Meteo
│   ├── build_dataset.py        # Builds data/processed/ from the raw downloads
│   ├── evaluate_baselines.py   # Scores the baselines on the validation folds
│   ├── evaluate_linear_gam.py  # Fits and validates the linear GAM, writes its forecast
│   ├── check_run_availability.py  # Publication time of every forecast run used
│   ├── check_station_break.py  # Evidence for the July 2025 break in RDU's readings
│   ├── break_sensitivity.py    # Scores under each treatment of that break
│   └── score_submission.py     # Scores final forecasts against Sep 17-30
├── src/weather_modeling/       # Reusable code imported by notebooks/scripts
├── tests/                      # Unit tests
├── models/                     # Trained models / serialized artifacts
├── reports/figures/            # Figures for the write-up and presentation
├── reports/predictions/        # Final forecasts, one CSV per model
├── docs/                       # Project documentation
│   ├── README_part1.md         # What was built for the data and evaluation part
│   ├── data_and_evaluation.md  # Data dictionary and how to evaluate a model
│   └── linear_gam.md           # The linear GAM: approach, inputs, how to run
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate.ps1
python -m pip install --upgrade pip   # older pip cannot do the editable install below
pip install -r requirements.txt
pip install -e .                 # makes `weather_modeling` importable in notebooks
```

## Data

Source: NOAA NCEI Global Historical Climatology Network hourly (GHCNh),
station `USW00013722` (Raleigh-Durham International Airport).

Regenerate the raw data (standard library only, no install needed):

```bash
python scripts/download_data.py
```

The script keeps routine hourly reports (FM15, minute 51), converts timestamps to
US Eastern, drops columns that are empty for the whole period, and writes one CSV
per local year into `data/raw/ghcnh_rdu/`.

Forecast runs come from the Open-Meteo Single Runs API (weather data by
Open-Meteo.com, CC BY 4.0), which returns each run exactly as it was issued.

`data/processed/forecast_pairs.parquet` is the model-ready table: one row per
(12z forecast run, target hour), laid out like the real task. Load it with
`weather_modeling.dataset.load_training_data()` and
`load_forecast_features()`. See [docs/data_and_evaluation.md](docs/data_and_evaluation.md)
for the columns, the validation folds, the baselines and how to score a model.

## Conventions

- Never edit files in `data/raw/`; write cleaned outputs to `data/processed/`.
- Name notebooks `NN_short_description.ipynb` and keep them in run order.
- Move logic reused across notebooks into `src/weather_modeling/`.
- Split by forecast run, never by row, and score every model with
  `weather_modeling.evaluation` so results are comparable.
- `data/raw/ghcnh_rdu_holdout/` is the test set: use it only through
  `scripts/score_submission.py`, once the models are final.
