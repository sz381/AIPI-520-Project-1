# Linear GAM

A generalised additive model with an identity link (`pygam.LinearGAM`), built
the same way as the linear regression in `notebooks/04_linear_regression_modeling_new_new.ipynb`
and trained on the same table from `03`
(`data/processed/model_samples_2024-03-14_to_2026-09-16.csv.gz`).

| File | Role |
|---|---|
| `scripts/linear_gam.py` | Fits, validates, saves the final models, makes and freezes the forecast |
| `scripts/linear_gam_test_evaluation.py` | Scores the frozen forecast on Sep 17-30, once |
| `notebooks/06_linear_gam.ipynb` | Reads those outputs; plots and comparisons only |

## Same approach as the linear regression

| | Linear regression (04) | Linear GAM |
|---|---|---|
| Target | temperature − `hist_avg` | temperature − `hist_avg` |
| Forecast | `hist_avg` + predicted difference | `hist_avg` + predicted difference |
| Models | one per forecast day (1-14) | one per forecast day (1-14) |
| Training weights | half-life 180 days | half-life 180 days |
| Inputs | the 12 `FEATURES` of 03 | the same information (below) |
| Form | straight line per input | smooth curve per input |

| Input | GAM term |
|---|---|
| `nwp_lagged_anomaly`, `nwp_anomaly` (ECMWF minus `hist_avg`) | spline |
| `nwp_recent_error` (ECMWF error over the last 6 observed hours) | spline |
| `anom_now`, `anom_24h` (station anomaly at the forecast start) | spline |
| `nwp_bias_14d`, `_30d`, `_60d` (recent ECMWF bias) | linear |
| hour of the target (`hour_sin1`..`hour_cos2` in the LR) | one cyclic spline on the local hour, 0-24 |

Splines have 10 basis functions (12 for the hour) and a second-difference
penalty. One smoothing value per forecast day, applied to all terms, is chosen
by GCV from `LAM_GRID`.

Validation is the same as in `04`: the two season-matched folds and the six
2026 windows, each trained only on targets before the period it is scored on,
with every method scored on the same hours. The script refits the `04` linear
regression next to the GAM (it reproduces the frozen `04` forecast exactly), so
"Linear regression" in the outputs is the `04` model.

## Running it

```bash
pip install -r requirements.txt                  # includes pygam
python scripts/linear_gam.py                     # a few minutes
python scripts/linear_gam_test_evaluation.py     # once, after the forecast is frozen
```

Then open `notebooks/06_linear_gam.ipynb` for the figures (`reports/figures/06_*.png`).

`linear_gam.py` writes `scripts/results/linear_gam_*.csv`, `models/linear_gam.pkl`
(not tracked by git), and the forecast
`reports/forecast_2026-09-17_to_2026-09-30_linear_gam.csv` with its SHA-256 in
`scripts/results/final_forecast_linear_gam.sha256`. If a frozen forecast with a
different hash already exists, the script stops instead of overwriting it
(`--refreeze` overrides, only before any test scoring).
`linear_gam_test_evaluation.py` checks that hash, then scores every method,
including the frozen `04` forecast, on the observed hours of
`data/external/rdu_obs_2026-09-17_to_2026-09-30.csv` (downloaded by `05`).
