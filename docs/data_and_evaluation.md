# Data and evaluation framework

Everything a model needs comes from three calls, and every model is scored the
same way:

```python
from weather_modeling.dataset import load_training_data, load_forecast_features, TARGET
from weather_modeling.evaluation import BASELINES, cross_validate, score_folds, make_submission

data = load_training_data(models=("ecmwf",))   # rows with a target, for fitting and validation
future = load_forecast_features()              # the 336 rows to predict (Sep 17-30)
```

## Pipeline

```bash
python scripts/download_data.py        # RDU observations 2015 .. Sep 16 2026 (already in the repo)
python scripts/download_nwp.py         # ECMWF and GFS 12z forecast runs (already in the repo)
python scripts/build_dataset.py        # -> data/processed/*.parquet
python scripts/evaluate_baselines.py   # -> reports/baselines_*.csv, reports/figures/
python scripts/break_sensitivity.py    # -> reports/station_break_sensitivity.csv
python scripts/check_run_availability.py   # -> reports/run_availability.csv
pytest                                 # geometry, alignment, quality and leakage checks
```

| File | Content |
|---|---|
| `data/raw/ghcnh_rdu/` | Hourly observations (NOAA GHCNh), up to Sep 16 23:51 local |
| `data/raw/openmeteo_single_runs/<model>/` | Archived forecast runs, one row per (run, forecast hour) |
| `data/raw/ghcnh_rdu_holdout/` | Observations of Sep 17-30. Final scoring only |
| `data/processed/obs_hourly.parquet` | All yearly raw files combined: one row per hourly report, as reported, with quality codes |
| `data/processed/climatology.parquet` | 2015-2023 mean temperature by day of year and UTC hour |
| `data/processed/forecast_pairs.parquet` | The model-ready table described below |

## What may be used

The cutoff is Sep 17 2026 00:00 EDT, which is 04:00 UTC.

| Data | Allowed? | In this repo |
|---|---|---|
| Observations reported before the cutoff | Yes | `data/raw/ghcnh_rdu/`, ending with the 23:51 EDT report of Sep 16 |
| GFS/ECMWF forecasts published before the cutoff | Yes | The Sep 16 12z runs, and earlier runs for training |
| Observations from after the cutoff | Only to score frozen forecasts | `data/raw/ghcnh_rdu_holdout/`, read only by `score_submission.py` |
| Forecast runs published after the cutoff | No | `nwp.py` refuses any run after Sep 16 12z |
| Reanalysis such as ERA5 | Not as a stand-in for a forecast | Used only to diagnose the station break; never a feature or a target |

### Initialisation, publication and valid time

A forecast has three times: when the run is initialised, when it is published,
and the hour it is valid for. The Single Runs archive is keyed by
initialisation time, which does not by itself show that a run was published
before a cutoff. The pipeline's assumption is explicit: **a run is available 16
hours after initialisation.** For the real forecast that is exactly the cutoff.

`scripts/check_run_availability.py` checks the assumption against the upload
time of each run's last needed file on the public storage of NOAA (GFS) and
ECMWF (open data); `reports/run_availability.csv` has one line per run.

| | GFS | ECMWF |
|---|---|---|
| Runs used | 167 | 910 |
| Published after initialisation, median | 5.1 h | 7.9 h |
| Published after initialisation, latest | 5.6 h | 14.8 h |
| The Sep 16 12z run | 17:09 UTC, 10.8 h before the cutoff | 19:34 UTC, 8.4 h before the cutoff |

- **Two ECMWF runs are excluded** (2024-09-17 and 2025-02-24, `EXCLUDED_RUNS` in
  `nwp.py`): their uploads are dated 37 and 24 hours after initialisation. An
  upload time can only overstate the delay, so this does not prove they were
  late, but nothing shows they were on time.
- **A gap in the evidence**: for ECMWF runs from 2024-07-16 to 2024-11-11 the
  public files stop at 240 hours, so the upload time shown is that of the
  240-hour file. The archive's values beyond 240 hours for those 119 runs come
  from a product whose publication time is not recorded here.
- The Sep 17 00z runs are initialised four hours before the cutoff but
  published after it, so they are not used.

### Scoring truth

The truth is the routine hourly report of NOAA GHCNh station `USW00013722`
(RDU), the one made at minute 51 of each hour. With the convention below, label
Sep 17 00:00 is the 00:51 report. These reports were made after the cutoff:
they are read only to score forecasts that are already frozen. A missing hour
is not filled in; `score_submission.py` prints how many of the 336 hours were
scored and their quality classes.

## How one forecast is laid out

The real forecast is issued at the cutoff from the ECMWF and GFS runs
initialised on Sep 16 at 12:00 UTC.

Every historical 12z run is treated as if it were that real forecast:

```
run_time (12:00 UTC)      cutoff = run + 16h           run + 17h ... run + 352h
        |---- observed, usable ----|---- lead day 1 ----| ... |---- lead day 14 ----|
```

A row's features use only the run itself, observations reported before that
run's cutoff, and the 2015-2023 climatology. `tests/test_dataset.py` rebuilds
runs from observations truncated at the cutoff and checks nothing changes.

### Which report is "12am"

Everything is kept in UTC; `target_time_local` is the only local-time column.

**The project's convention: a report belongs to the clock hour it was taken
in.** The report at 00:51 on Sep 17 is labelled Sep 17 00:00.

- The first target is therefore observed after the cutoff: it is an unknown
  future observation, like the other 335.
- Every observation used for training or as a feature was taken before Sep 17
  00:00 Eastern; the last one is the 23:51 report of Sep 16. Later observations
  are used only for scoring.
- Forecasts are on the full hour, so a report is paired with the forecast nine
  minutes later: the 00:51 report with the 01:00 forecast, not the 00:00 one.
  That forecast comes from a run that was available before the cutoff.

This is the only convention the code implements. The other possible reading,
taking the report closest to the hour (23:51 the evening before as "12am"), was
considered and dropped: it would make the first target hour already observed at
the cutoff. The choice was made on the definition, not by comparing scores.

## Columns of the pairs table

One row per (run, target hour): 917 runs x 336 hours. Temperatures in degC.

| Column | Meaning |
|---|---|
| `run_time_utc` | Initialisation time of the 12z run |
| `lead_hour` | Hours from run time to valid time, 17..352 |
| `lead_day` | 1..14; day 1 is the first 24 target hours |
| `valid_time_utc`, `target_time_local` | When the row verifies (see above) |
| `hour_utc`, `hour_local`, `doy` | Calendar features (`doy` on a 365-day calendar) |
| `obs_temp_c` | **Target**, as reported. Missing after the cutoff, for unreported hours and for rejected reports |
| `target_quality` | Quality class of the target report (see below). Not a feature |
| `target_before_break` | True where the target was measured before the July 2025 break (see below). Not a feature |
| `clim_temp_c` | Climatology at the valid time |
| `persist_temp_c` | Observation at the same clock hour in the last 24h before the cutoff |
| `last_obs_temp_c` | Last observation before the cutoff |
| `anom_24h`, `anom_7d`, `anom_30d` | Mean (observed - climatology) over that period before the cutoff |
| `<model>_t2m`, `_td2m`, `_rh2m`, `_cloud`, `_wind`, `_wdir`, `_precip`, `_swrad`, `_mslp` | The run's forecast at the valid time: temperature, dew point (degC), humidity (%), cloud cover (%), wind speed (m/s) and direction, precipitation (mm), shortwave radiation (W/m2), sea-level pressure (hPa). `<model>` is `ecmwf` or `gfs` |
| `<model>_bias_0_16h` | Mean (forecast - observed) over the run's first 16 hours, already observed at the cutoff |

`weather_modeling.features.add_fourier_features` adds sin/cos terms for the
hour and the day of year.

## Observation quality

A value being present does not make it good. `obs_hourly.parquet` keeps the
GHCNh quality code of each variable (`temp_qc`, `dewpoint_qc`, ...) and the
source code of the temperature (`temp_source`). The meaning of a code depends
on the source (GHCNh documentation, section VI), so `temp_quality` translates
the pair into one class:

| Class | Source and code | Reports | Treatment |
|---|---|---|---|
| `passed` | 343: 1, 5. 223: 1 | 98,256 | Used |
| `accepted` | 343: 0, 4, 9 (gross limits only), A (suspect, accepted as good), P, U, I, M, R (edited by a validator). 223: 4 (calculated) | 39 | Used |
| `unverified` | No code, or 223: 0 (not checked) | 4,168 | Used, after a plausibility check |
| `rejected` | 343: 2, 6 (suspect), 3, 7 (erroneous). 223: 2, 3, 5 (removed). Or unverified and implausible | 8 | `obs_temp_c` set to missing |

- **Every report since 2026-03-27 is unverified.** From that date the archive
  takes RDU from source 413, whose reports carry no quality code and which the
  documentation's code tables do not cover. That is all validation targets of
  the `late_summer_2026` fold and the whole target window, so the scores on both
  rest on data nobody has quality-checked. The `autumn_2025` fold's targets all
  passed.
- **The plausibility check** rejects an unverified reading outside -35..50 degC,
  or more than 8 degC away from both neighbouring hours in the same direction.
  None of the 4,168 fails it.
- **The 8 rejected reports** are from Dec 2024 to Mar 2025. They are never a
  target and never enter a feature.
- **Only the temperature is screened.** The other four variables keep their
  code but are not used by any feature; check the code before using them.

## Things to know about the data

- **GFS is missing by construction, not at random.** The table starts in 2024
  and the GFS archive starts on 2026-04-02, so `gfs_*` is missing in 82% of
  rows. Never fill those in. Keep two datasets instead:
  `load_training_data(models=("ecmwf",))` (286,693 rows, 909 runs) and
  `load_training_data(models=("ecmwf", "gfs"))` (53,148 rows, 165 runs).
- **The climatology uses 2015-2023, not 2015-2025.** With 2025 included, the
  validation targets of the `autumn_2025` fold would be part of their own
  climatology. The effect is small and goes both ways: with 2015-2025 the
  climatology baseline moves from 2.99 to 2.91 degC MAE on the 2025 fold and
  from 2.47 to 2.61 on the 2026 fold. `config.CLIM_LAST_YEAR` changes it.
- **ECMWF 12z runs reach 10 days until 2024-07-15** and 15 days after, so lead
  days 11-14 have no ECMWF forecast before then.
- **A few runs are missing from the archive** (ECMWF 2025-08-04/05/06/08 and
  2026-06-10; GFS 2026-09-14), two are cut short (ECMWF 2026-08-26 after 72h,
  GFS 2026-06-10 after 48h), and two ECMWF runs are excluded for lack of proof
  of timely publication (see above). Their rows are present with missing
  forecasts.
- **Other missing values are rare** (under 1% of trainable rows): hours with no
  observation, and occasional gaps in forecast wind or precipitation. Linear
  models need to drop or fill them.
- **Forecasts are smoother at long leads.** ECMWF is hourly up to 90h, 3-hourly
  up to 144h and 6-hourly beyond; GFS is hourly up to 120h and 3-hourly beyond.
  Open-Meteo interpolates the rest to hourly values, which flattens the daily
  maximum and minimum.

## The break in the station record, July 2025

Against both ECMWF forecasts and the ERA5 reanalysis, the reported temperature
steps up in early 2024 and drops back around 2025-07-22
(`python scripts/check_station_break.py`, `reports/figures/station_break.png`).

| Evidence | Size of the drop |
|---|---|
| Against ECMWF forecasts, 12 months either side | 1.0 degC |
| Against ERA5, 12 months either side | 0.7 degC |
| Against Greensboro, Rocky Mount, Fayetteville (one-off check, not in the scripts) | 1.0, 0.8, 0.5 degC |
| Within every combination of day/night, cloud cover and temperature band (one-off) | 0.5 to 1.4 degC |
| Within a single data source (343), before and after | 1.5 degC |

So it is not a change in the weather, and not the archive's switch of data
source on 2025-08-27. It shows the two periods were measured differently. It
does not show which one is right, or that either is wrong. NWS announced the
replacement of ASOS temperature sensors nationwide during 2025, which makes a
sensor change the likely cause, but that confirms neither the date at RDU nor
the size of the offset. The start of the higher period, early 2024, is less
certain than its end.

**The primary analysis uses every reading as reported.** Nothing is shifted in
`obs_hourly.parquet` or `forecast_pairs.parquet`; `target_before_break` only
marks the rows. `evaluation.break_sensitivity` scores two alternatives next to
it, with no information from the future:

- **shifted**: readings before the break are lowered by an offset estimated only
  from data available at the fold's first cutoff
  (`station_break.estimate_offset`);
- **dropped**: targets from before the break are left out of training.

14-day MAE in degC. `ecmwf_lr` is a reference regression per lead day on
`ecmwf_t2m`, `clim_temp_c` and `anom_24h`, defined in
`scripts/break_sensitivity.py`:

| Fold | Treatment | Offset | From | `station_lr` | `ecmwf_lr` | `raw_ecmwf` |
|---|---|---|---|---|---|---|
| `autumn_2025` | as reported | | | 3.16 | 2.69 | 2.45 |
| `autumn_2025` | shifted | 1.49 | 25 days | 2.81 | 2.24 | 2.45 |
| `autumn_2025` | dropped | | | too few runs (35) | | |
| `late_summer_2026` | as reported | | | 2.16 | 1.82 | 2.10 |
| `late_summer_2026` | shifted | 0.96 | 365 days | 2.28 | 1.82 | 2.10 |
| `late_summer_2026` | dropped | | | 2.31 | 1.83 | 2.10 |

- On the 2026 fold, which resembles the real forecast (a year of post-break
  data in training), the treatment makes no difference to `ecmwf_lr` and
  as-reported is best for `station_lr`.
- On the 2025 fold, which starts 25 days after the break, a regression trained
  on as-reported data is worse than raw ECMWF, because nearly all its training
  targets come from before the break. Shifting repairs that.
- What remains hindsight even in the shifted treatment: the date of the break
  was identified with data up to 2026. Only the size of the offset is estimated
  as of each fold.
- A 25-day estimate does not cancel the seasonal cycle of the forecast error;
  a full year does (see `estimate_offset`).

## Evaluating a model

A model is a function that fits on a training table and predicts a validation
table. The validation table is passed without the target.

```python
from sklearn.linear_model import LinearRegression

FEATURES = ["ecmwf_t2m", "clim_temp_c", "anom_24h"]

def fit_predict(train, val):
    train = train.dropna(subset=FEATURES)
    model = LinearRegression().fit(train[FEATURES], train[TARGET])
    return model.predict(val[FEATURES])

data = load_training_data(models=("ecmwf",)).dropna(subset=FEATURES)
scored = cross_validate({"my_lr": fit_predict}, data)       # out-of-fold predictions + baselines
score_folds(scored, [*BASELINES, "my_lr"], metric="mae")    # table by fold and lead day
```

- **Folds** (`splits.FOLDS`): `autumn_2025` validates on runs Aug 15 - Oct 15
  2025, `late_summer_2026` on runs Aug 1 - Sep 16 2026. Each trains on all
  earlier runs. Only the 2026 fold has GFS. `splits.monthly_folds` gives one
  fold per month for tuning.
- **Never split rows at random.** Rows of one run, and of neighbouring runs,
  describe the same weather. `split_by_run` is purged: a training row is kept
  only if its target time comes before the first validation target.
- **Baselines**: `persistence`, `climatology`, `station_lr` (a linear
  regression on climatology and the 24h anomaly, fitted per lead day on the
  fold's training rows: the best forecast without any NWP model), `raw_ecmwf`,
  `raw_gfs`.
- `evaluation.break_sensitivity({"my_lr": fit_predict})` repeats the scoring
  under the three treatments of the station break.
- `evaluation.plot_by_lead_day(table.loc[fold], title, path=...)` draws the
  lead-day figure.

### How the scores are reported

The assignment does not prescribe a metric, so the choice is ours, and it has
to be made once and before the test scores are looked at. The framework's
default is **MAE in degC with the 14 lead days weighted equally**, with RMSE and
bias alongside. The same forecasts give different numbers under different
conventions, so a score is comparable only with one reported the same way.

- **Unit.** Errors are in degC. `fahrenheit=True` gives degF, which is 1.8
  times larger.
- **Weighting.** The `all` row weights the 14 lead days equally by default, as
  in the real task, where each day has 24 hours. `overall="pooled"` weights
  every available hour equally instead. On validation the two differ, because
  late runs have fewer verified long-lead hours.

Raw ECMWF on the 2026 fold, for example: 2.10 degC (lead days equal), 2.02 degC
(hours pooled), 3.78 degF and 3.64 degF. `reports/baselines_overall.csv` has
all four for every baseline.

- **Metrics** (`metrics.score_by`): `mae`, `rmse`, `bias` (predicted -
  observed), by lead day or any other column. All models in one call are scored
  on the same rows.

### How much a fold can tell

A fold is 46 or 62 daily runs, but each run's 14-day forecast overlaps its
neighbours', so these are far from 46 or 62 independent samples.
`metrics.bootstrap_mae_difference(scored, "a", "b")` resamples blocks of 14
days of runs to cover that overlap. A fold then holds only four or five blocks,
so the interval is a rough guide and small differences between models should
be read as noise.

## The final forecast

```python
final_model_predictions = ...   # 336 values, in the row order of load_forecast_features()
make_submission(final_model_predictions, "my_lr")   # -> reports/predictions/my_lr.csv
```

```bash
python scripts/download_data.py --holdout
python scripts/score_submission.py reports/predictions/my_lr.csv reports/predictions/my_gbm.csv
```

`score_submission.py` compares the submissions and the baselines with the
observed temperatures of Sep 17-30. That is the test set: run it once the
forecasts are frozen, and do not tune against it. Each scoring is appended to
`reports/final_scoring_log.csv` with a SHA-256 fingerprint of the forecast file,
and the script warns if a file has changed since it was first scored. NOAA
publishes with a delay; on Oct 4
the file ended at Sep 30 09:51 local (322 of 336 hours, all unverified), so
re-run the download before the final numbers.
