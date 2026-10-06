# Linear GAM

A generalised additive model with an identity link (`pygam.LinearGAM`), built
the same way as the linear-regression baseline. Code: `src/weather_modeling/gam.py`. Runner:
`scripts/evaluate_linear_gam.py`. Notebook: `notebooks/06_linear_gam.ipynb`.

## Same approach as the LR

| | LR baseline | Linear GAM |
|---|---|---|
| Target | observed − historical average | observed − `clim_temp_c` |
| Forecast | historical average + predicted anomaly | `clim_temp_c` + predicted anomaly |
| Models | one per lead day (1-14) | one per lead day (1-14) |
| Training weights | half-life 180 days | half-life 180 days |
| Inputs | 12 (below) | the same 12, hour as one cyclic term |
| Form | straight line per input | smooth curve per input |

Inputs (`gam.LR_FEATURES`), each known at the run's cutoff:

| Input | Meaning | GAM term |
|---|---|---|
| `nwp_anomaly` | ECMWF 12z forecast − climatology | spline |
| `nwp_lagged_anomaly` | mean of the 12z runs of days d, d−1, d−2 − climatology | spline |
| `nwp_recent_error` | observed − forecast over the run's last 6 observed hours | spline |
| `nwp_bias_14d`, `_30d`, `_60d` | mean observed − forecast at lead day 1 over the previous 14/30/60 days | linear |
| `anom_now` | observed − climatology at the cutoff | spline |
| `anom_24h` | mean observed − climatology over the last 24 h | spline |
| `hour_sin1..cos2` (LR) / `hour_local` (GAM) | target hour | cyclic spline (0-24) |

Splines: 10 basis functions (12 for the hour), second-difference penalty. The
smoothing parameter is one value per lead day, applied to all terms, chosen by
GCV from `gam.LAM_GRID`.

`lr_mos` is the LR rebuilt on the same rows and inputs (sklearn
`LinearRegression`), so GAM vs `lr_mos` isolates the effect of smooth terms.

## Differences from the LR's own pipeline

- **Climatology.** The anomaly is taken against `clim_temp_c` (2015-2023,
  ±10 days, by UTC hour), not the LR notebook's expanding "earlier years"
  average. This is what the shared harness and baselines use.
- **Lagged ensemble.** This repo archives 12z runs only, so the lagged mean
  uses the 12z runs of days d, d−1 and d−2 instead of 12z d, 00z d and 12z d−1.
- **Timing conventions** are the repo's (`config.py`): run + 16 h cutoff, the
  xx:51 report paired with the next full hour.

With these changes `lr_mos` reproduces the LR baseline closely: 14-day MAE
(hours pooled) 2.24 / 1.77 °C on the 2025 / 2026 folds against the PR's
2.20 / 1.78.

## Checks

`tests/test_gam.py`: features are unchanged when a run is rebuilt from
observations truncated at its cutoff and runs up to itself (four runs, including
the issue run); the 336 final rows have every input; a smoke fit of the GAM
(skipped if pygam is not installed).

## Running it

```bash
pip install -r requirements.txt     # adds pygam
python scripts/build_dataset.py     # if data/processed/ is missing
python scripts/evaluate_linear_gam.py
pytest tests/test_gam.py
# once, after the forecasts are frozen:
python scripts/score_submission.py reports/predictions/linear_gam.csv reports/predictions/lr_mos.csv
```

The evaluation fits 14 GAMs per fold, for 2 folds, 6 windows and the final
model, with a 7-value lambda search each; expect a few minutes.
