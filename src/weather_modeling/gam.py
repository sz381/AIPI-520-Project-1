"""Linear GAM (MOS) for the RDU forecast, on the same inputs as the LR baseline.

- The target is the **anomaly**: observed temperature minus climatology
  (``clim_temp_c``). The forecast is climatology + predicted anomaly.
- **One model per lead day** (1-14), so each day learns how far to trust ECMWF.
- Training rows are **weighted by recency** (half-life 180 days) because the
  ECMWF bias at RDU drifts.
- The **inputs are the LR's 12 features** (see ``LR_FEATURES``), built only
  from information available at each run's cutoff.

The difference is the functional form. The LR adds up straight lines; the GAM
adds up smooth curves, one per input (penalised B-splines, identity link,
``pygam.LinearGAM``). The hour of day is a single cyclic spline instead of four
sin/cos terms, and the three slow-moving ECMWF bias terms stay linear.

Typical use::

    table = build_mos_table()                       # pairs table + the MOS inputs
    data = trainable(table)
    scored = cross_validate({"linear_gam": gam_fit_predict, "lr_mos": lr_fit_predict}, data)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config
from .climatology import climatology_at, load_climatology
from .dataset import TARGET, load_pairs
from .nwp import load_nwp
from .obs import load_obs, temperature_series


# Recent runs count more: a training run's weight halves every HALF_LIFE_DAYS
# before the newest training run (same as the LR baseline).
HALF_LIFE_DAYS = 180

# The LR baseline's 12 inputs, named as in its notebook 03.
LR_FEATURES = [
    "nwp_lagged_anomaly",  # mean of the last 12z runs at the valid time, minus climatology
    "nwp_anomaly",         # this run's ECMWF forecast minus climatology
    "nwp_recent_error",    # observed - forecast over the run's last 6 observed hours
    "nwp_bias_14d",        # mean observed - forecast over the previous 14 / 30 / 60 days
    "nwp_bias_30d",
    "nwp_bias_60d",
    "anom_now",            # observed - climatology at the cutoff (last report)
    "anom_24h",            # mean observed - climatology over the 24h before the cutoff
    "hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2",  # local hour of the target
]

# The GAM's terms. Same information as LR_FEATURES; the hour enters once, as a
# cyclic spline on the local clock hour (0-23, with 23 next to 0).
GAM_SMOOTH = ["nwp_lagged_anomaly", "nwp_anomaly", "nwp_recent_error", "anom_now", "anom_24h"]
GAM_LINEAR = ["nwp_bias_14d", "nwp_bias_30d", "nwp_bias_60d"]
GAM_HOUR = "hour_local"
GAM_FEATURES = [*GAM_SMOOTH, *GAM_LINEAR, GAM_HOUR]

# Spline settings and the lambda grid searched by GCV for each lead day. One
# lambda value is applied to all terms at a time.
N_SPLINES = 10
N_SPLINES_HOUR = 12
LAM_GRID = np.logspace(-2, 4, 7)

# Members of the time-lagged ensemble: this 12z run and the two before it.
# (The LR also used the 00z run of the same day; this repo archives 12z runs only.)
LAGGED_RUN_OFFSETS_DAYS = (0, 1, 2)
# The run's "recent error" covers its last 6 observed hours: leads 11..16
# (6pm-11pm Eastern daylight time on the run day).
RECENT_ERROR_LEADS = (config.OBS_KNOWN_LEAD - 5, config.OBS_KNOWN_LEAD)
# Daily ECMWF error: leads 17..40 (lead day 1) of each run. Run d-2 is fully
# verified 8 hours before run d's cutoff, hence the shift of 2 days.
BIAS_LEADS = (config.FIRST_LEAD, config.FIRST_LEAD + 23)
BIAS_SHIFT_DAYS = 2
BIAS_WINDOWS = (14, 30, 60)

# Six extra 14-day forecasts issued like the real one (the 12z run of the day
# before, covering 12am day 1 .. 11pm day 14 local), as in the LR baseline's
# extra check. Fold name -> (first, last) run date for ``evaluation.cross_validate``.
TASK_WINDOWS_2026 = {
    name: (pd.Timestamp(first) - pd.Timedelta(days=1),) * 2
    for name, first in {
        "Jun 1-14": "2026-06-01", "Jun 17-30": "2026-06-17", "Jul 1-14": "2026-07-01",
        "Jul 17-30": "2026-07-17", "Aug 1-14": "2026-08-01", "Aug 17-30": "2026-08-17",
    }.items()
}


# --------------------------------------------------------------------------- features


def _forecast_lookup(forecasts):
    return forecasts.set_index(["run_time_utc", "valid_time_utc"])["t2m"].sort_index()


def _lookup(series, runs, valid):
    index = pd.MultiIndex.from_arrays([pd.DatetimeIndex(runs), pd.DatetimeIndex(valid)])
    return series.reindex(index).to_numpy()


def run_features(temp, clim, forecasts, runs):
    """Features that are the same for every row of a run, indexed by ``run_time_utc``.

    ``temp`` is the hourly temperature series (``obs.temperature_series``) and
    ``forecasts`` the ECMWF runs (``nwp.load_nwp("ecmwf")``). Only observations
    up to each run's cutoff (run + 16h) and runs up to that run are used.
    """
    runs = pd.DatetimeIndex(runs)
    known = runs + pd.Timedelta(hours=config.OBS_KNOWN_LEAD)
    anomaly = temp - climatology_at(clim, temp.index)
    out = pd.DataFrame(index=runs)
    out.index.name = "run_time_utc"
    out["anom_now"] = anomaly.ffill(limit=3).reindex(known).to_numpy()

    # How wrong this run already is: observed - forecast over its last 6 observed hours.
    first, last = RECENT_ERROR_LEADS
    early = forecasts[forecasts["lead_hour"].between(first, last) & forecasts["run_time_utc"].isin(runs)]
    error = temp.reindex(early["valid_time_utc"]).to_numpy() - early["t2m"].to_numpy()
    out["nwp_recent_error"] = pd.Series(error, index=early["run_time_utc"].to_numpy()).groupby(level=0).mean()

    # Recent systematic bias: daily mean(observed - forecast) at lead day 1, from runs
    # 2 days old and older, averaged over the last 14 / 30 / 60 days.
    first, last = BIAS_LEADS
    verify = forecasts[forecasts["lead_hour"].between(first, last)]
    error = temp.reindex(verify["valid_time_utc"]).to_numpy() - verify["t2m"].to_numpy()
    daily = pd.Series(error, index=verify["run_time_utc"].to_numpy()).groupby(level=0).mean()
    days = pd.date_range(min(daily.index.min(), runs.min()), max(daily.index.max(), runs.max()), freq="D")
    daily = daily.reindex(days)
    for window in BIAS_WINDOWS:
        rolled = daily.shift(BIAS_SHIFT_DAYS).rolling(window, min_periods=window // 2).mean()
        out[f"nwp_bias_{window}d"] = rolled.reindex(runs).to_numpy()
    return out


def add_mos_features(pairs, temp, clim, forecasts):
    """Add the LR's inputs (``LR_FEATURES``) and the anomaly target to the pairs table.

    Gaps are filled as in the LR baseline, only with information from the same
    moment: the current anomaly from the 24h / 7-day anomaly, the recent error
    from the 30-day bias, and anything still missing with 0 ("as normal", "no
    error"). Rows without an ECMWF forecast keep NaN in the NWP features.
    """
    out = pairs.merge(run_features(temp, clim, forecasts, pairs["run_time_utc"].unique()),
                      left_on="run_time_utc", right_index=True, how="left")

    # Time-lagged ensemble of the 12z runs at the same valid time.
    lookup = _forecast_lookup(forecasts)
    members = [
        _lookup(lookup, out["run_time_utc"] - pd.Timedelta(days=days), out["valid_time_utc"])
        for days in LAGGED_RUN_OFFSETS_DAYS
    ]
    # Mean of the members that exist (older runs end before the last lead hours).
    lagged = pd.DataFrame(np.column_stack(members), index=out.index).mean(axis=1)
    out["ecmwf_lagged_mean"] = lagged.where(out["ecmwf_t2m"].notna())

    out["anom_now"] = out["anom_now"].fillna(out["anom_24h"]).fillna(out["anom_7d"]).fillna(0.0)
    out["anom_24h"] = out["anom_24h"].fillna(out["anom_now"])
    out["nwp_recent_error"] = out["nwp_recent_error"].fillna(out["nwp_bias_30d"]).fillna(0.0)
    for window in BIAS_WINDOWS:
        out[f"nwp_bias_{window}d"] = out[f"nwp_bias_{window}d"].fillna(0.0)

    out["nwp_anomaly"] = out["ecmwf_t2m"] - out["clim_temp_c"]
    out["nwp_lagged_anomaly"] = out["ecmwf_lagged_mean"] - out["clim_temp_c"]
    for k in (1, 2):
        angle = 2 * np.pi * k * out["hour_local"] / 24
        out[f"hour_sin{k}"], out[f"hour_cos{k}"] = np.sin(angle), np.cos(angle)
    out["target_anomaly"] = out[TARGET] - out["clim_temp_c"]
    return out


def build_mos_table(pairs=None, obs=None, clim=None, forecasts=None):
    """The pairs table (every run, including the issue run) with the MOS inputs added."""
    pairs = load_pairs() if pairs is None else pairs
    obs = load_obs() if obs is None else obs
    clim = load_climatology() if clim is None else clim
    forecasts = load_nwp("ecmwf") if forecasts is None else forecasts
    return add_mos_features(pairs, temperature_series(obs), clim, forecasts)


def trainable(table):
    """Rows with a target and an ECMWF forecast (as ``load_training_data(("ecmwf",))``)."""
    return table[table[TARGET].notna() & table["ecmwf_t2m"].notna()].reset_index(drop=True)


def forecast_rows(table):
    """The 336 rows to predict: the Sep 16 12z run."""
    return table[table["run_time_utc"] == config.ISSUE_RUN].reset_index(drop=True)


# --------------------------------------------------------------------------- models


def recency_weights(runs, newest=None, half_life_days=HALF_LIFE_DAYS):
    runs = pd.to_datetime(runs)
    newest = runs.max() if newest is None else newest
    age_days = (newest - runs).dt.total_seconds() / 86400
    return (0.5 ** (age_days / half_life_days)).to_numpy()


def _fit_rows(train, features, lead_day):
    rows = train[train["lead_day"] == lead_day].dropna(subset=[TARGET, "clim_temp_c", *features])
    return rows, rows[TARGET] - rows["clim_temp_c"]


def gam_terms():
    """pygam terms in the order of ``GAM_FEATURES``: smooths, linear terms, cyclic hour."""
    from pygam import l, s

    terms = None
    for i, _ in enumerate(GAM_SMOOTH):
        term = s(i, n_splines=N_SPLINES)
        terms = term if terms is None else terms + term
    offset = len(GAM_SMOOTH)
    for i, _ in enumerate(GAM_LINEAR):
        terms = terms + l(offset + i)
    hour_index = GAM_FEATURES.index(GAM_HOUR)
    terms = terms + s(hour_index, n_splines=N_SPLINES_HOUR, basis="cp", edge_knots=np.array([0.0, 24.0]))
    return terms


def fit_gam(train, lam_grid=LAM_GRID, half_life_days=HALF_LIFE_DAYS):
    """One ``pygam.LinearGAM`` per lead day on the anomaly; lambda chosen by GCV per lead day."""
    from pygam import LinearGAM

    newest = train["run_time_utc"].max()
    models = {}
    for lead_day in range(1, config.N_LEAD_DAYS + 1):
        rows, y = _fit_rows(train, GAM_FEATURES, lead_day)
        weights = recency_weights(rows["run_time_utc"], newest, half_life_days)
        gam = LinearGAM(gam_terms())
        gam.gridsearch(rows[GAM_FEATURES].to_numpy(float), y.to_numpy(float), weights=weights,
                       lam=lam_grid, progress=False)
        models[lead_day] = gam
    return models


def fit_lr(train, half_life_days=HALF_LIFE_DAYS):
    """The LR baseline on the same rows and features: one LinearRegression per lead day."""
    newest = train["run_time_utc"].max()
    models = {}
    for lead_day in range(1, config.N_LEAD_DAYS + 1):
        rows, y = _fit_rows(train, LR_FEATURES, lead_day)
        weights = recency_weights(rows["run_time_utc"], newest, half_life_days)
        models[lead_day] = LinearRegression().fit(rows[LR_FEATURES].to_numpy(float), y.to_numpy(float),
                                                  sample_weight=weights)
    return models


def predict(models, rows, features):
    """Climatology + predicted anomaly; NaN where an input is missing."""
    predicted = np.full(len(rows), np.nan)
    complete = rows[features].notna().all(axis=1).to_numpy()
    for lead_day, model in models.items():
        ok = complete & (rows["lead_day"] == lead_day).to_numpy()
        if ok.any():
            anomaly = model.predict(rows.loc[ok, features].to_numpy(float))
            predicted[ok] = rows["clim_temp_c"].to_numpy()[ok] + anomaly
    return predicted


def gam_fit_predict(train, val):
    """``fit_predict`` for ``evaluation.cross_validate``."""
    return predict(fit_gam(train), val, GAM_FEATURES)


def lr_fit_predict(train, val):
    return predict(fit_lr(train), val, LR_FEATURES)


def ecmwf_weight(models, rows, lead_days=None):
    """How much of the ECMWF anomaly each lead day's GAM keeps.

    The slope of the GAM's response to a +1 degC shift of both ECMWF anomaly
    inputs, averaged over ``rows`` (the analogue of the LR's two ECMWF
    coefficients added up). 1 means "follow ECMWF", 0 means "stay on climatology".
    """
    weights = {}
    for lead_day, model in models.items():
        if lead_days is not None and lead_day not in lead_days:
            continue
        part = rows[rows["lead_day"] == lead_day].dropna(subset=GAM_FEATURES)
        base = part[GAM_FEATURES].to_numpy(float)
        bumped = base.copy()
        for name in ("nwp_anomaly", "nwp_lagged_anomaly"):
            bumped[:, GAM_FEATURES.index(name)] += 1.0
        weights[lead_day] = float(np.mean(model.predict(bumped) - model.predict(base)))
    return pd.Series(weights, name="weight_on_ecmwf")
