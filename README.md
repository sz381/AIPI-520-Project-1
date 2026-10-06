# AIPI 520 – Project 1: Modeling Weather

Modeling weather at Raleigh-Durham International Airport (RDU) using NOAA GHCNh
hourly observations, 2015-01-01 through 2026-09-16.

## Project structure

```
.
├── data/
│   ├── raw/ghcnh_rdu/<year>/   # Immutable downloads, one CSV per (Eastern) year
│   ├── processed/              # Final, model-ready datasets
├── notebooks/                  # Exploratory work, numbered in run order
│   ├── 01_data_sourcing.ipynb
│   └── 06_linear_gam.ipynb     # Plots and comparisons of the linear GAM's results
├── scripts/                    # Runnable, reproducible entry points
│   ├── download_data.py        # Downloads raw GHCNh data for RDU
│   ├── linear_gam.py           # Fits and validates the linear GAM, freezes its forecast
│   └── linear_gam_test_evaluation.py  # Scores the frozen GAM forecast on Sep 17-30
├── src/weather_modeling/       # Reusable code imported by notebooks/scripts
├── tests/                      # Unit tests
├── models/                     # Trained models / serialized artifacts
├── reports/figures/            # Figures for the write-up and presentation
├── docs/                       # Project documentation
│   └── linear_gam.md           # The linear GAM: approach, inputs, how to run
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate.ps1
pip install -r requirements.txt
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

`data/processed/rdu_ghcnh_2015-01-01_to_2026-09-15.csv` is a single combined file
of the raw data (through 2026-09-15).

## Conventions

- Never edit files in `data/raw/`; write cleaned outputs to `data/processed/`.
- Name notebooks `NN_short_description.ipynb` and keep them in run order.
- Move logic reused across notebooks into `src/weather_modeling/`.
