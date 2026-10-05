"""The model-ready table: one row per (12z forecast run, target hour).

Every historical 12z run is laid out exactly like the real task. The run is
initialised at ``run_time_utc``; the cutoff falls 16h later; the 336 target
hours (lead day 1..14) follow, starting at ``config.FIRST_LEAD``. A row's
features use only

- forecasts from that run (published well before the cutoff),
- observations reported before that run's cutoff,
- the 2015-2023 climatology,

so a row never contains information from at or after its own cutoff.
"""

import numpy as np
import pandas as pd

from . import config
from .climatology import climatology_at, day_of_year_365
from .nwp import VARIABLES
from .obs import temperature_series, to_local
from .station_break import before_break


TARGET = "obs_temp_c"

# Columns that describe the target rather than what is known at the cutoff.
TARGET_INFO = ["target_quality", "target_before_break"]

# Observation-based features, identical for all rows of a run.
ISSUE_FEATURES = ["last_obs_temp_c", "anom_24h", "anom_7d", "anom_30d"]


def nwp_columns(model):
    """Feature columns that come from one NWP model, e.g. ``ecmwf_t2m``."""
    return [f"{model}_{short}" for short in VARIABLES.values()] + [f"{model}_bias_0_16h"]


def issue_features(temp, clim, runs):
    """What the observations say at each run's cutoff (run + 16h).

    ``anom_*`` is the mean of (observed - climatology) over the trailing 24h,
    7 days and 30 days: how warm or cold the recent weather has been.
    """
    anomaly = temp - climatology_at(clim, temp.index)
    known = runs + pd.Timedelta(hours=config.OBS_KNOWN_LEAD)
    features = pd.DataFrame(index=runs)
    features["last_obs_temp_c"] = temp.ffill(limit=3).reindex(known).to_numpy()
    for hours, name in [(24, "anom_24h"), (168, "anom_7d"), (720, "anom_30d")]:
        features[name] = anomaly.rolling(hours, min_periods=hours // 2).mean().reindex(known).to_numpy()
    return features


def early_bias(forecasts, temp):
    """Mean (forecast - observed) temperature over lead 1..16h of each run.

    Those hours are already observed at the cutoff, so this is how wrong the
    run has been so far.
    """
    early = forecasts[forecasts["lead_hour"].between(1, config.OBS_KNOWN_LEAD)]
    error = early["t2m"].to_numpy() - temp.reindex(early["valid_time_utc"]).to_numpy()
    return pd.Series(error, index=early["run_time_utc"].to_numpy()).groupby(level=0).mean()


def build_pairs(obs, clim, forecasts, runs=None):
    """Build the table from observations, climatology and ``{model: forecasts}``.

    ``runs`` defaults to every daily 12z run from the first archived run to the
    issue run, so runs missing from the archive still get (forecast-less) rows.
    """
    if runs is None:
        first_run = min(frame["run_time_utc"].min() for frame in forecasts.values())
        runs = pd.date_range(first_run, config.ISSUE_RUN, freq="D")
    runs = pd.DatetimeIndex(runs)
    leads = np.arange(config.FIRST_LEAD, config.LAST_LEAD + 1)
    pairs = pd.MultiIndex.from_product([runs, leads], names=["run_time_utc", "lead_hour"]).to_frame(index=False)

    valid = pairs["run_time_utc"] + pd.to_timedelta(pairs["lead_hour"], unit="h")
    pairs["valid_time_utc"] = valid
    # The report with valid time H was taken at H-1:51, so it belongs to clock hour H-1.
    pairs["target_time_local"] = to_local(valid - pd.Timedelta(hours=1))
    pairs["lead_day"] = (pairs["lead_hour"] - config.FIRST_LEAD) // 24 + 1
    pairs["hour_local"] = pairs["target_time_local"].dt.hour
    pairs["hour_utc"] = valid.dt.hour
    pairs["doy"] = day_of_year_365(valid)

    temp = temperature_series(obs)
    pairs[TARGET] = temp.reindex(valid).to_numpy()
    # Target-side information, not features: the quality class of the report, and
    # whether it was measured before the July 2025 break in the station record.
    pairs["target_quality"] = obs.set_index("valid_time_utc")["temp_quality"].reindex(valid).to_numpy()
    pairs["target_before_break"] = before_break(valid)
    pairs["clim_temp_c"] = climatology_at(clim, valid)
    # Same clock hour within the last 24h before the cutoff.
    last_day = valid - pd.to_timedelta(24 * pairs["lead_day"], unit="h")
    pairs["persist_temp_c"] = temp.reindex(last_day).to_numpy()
    pairs = pairs.merge(issue_features(temp, clim, runs), left_on="run_time_utc", right_index=True, how="left")

    for model, frame in forecasts.items():
        renamed = frame.drop(columns="lead_hour").rename(
            columns={short: f"{model}_{short}" for short in VARIABLES.values()}
        )
        pairs = pairs.merge(renamed, on=["run_time_utc", "valid_time_utc"], how="left")
        pairs[f"{model}_bias_0_16h"] = pairs["run_time_utc"].map(early_bias(frame, temp))
    return pairs


def load_pairs():
    """Every row, including rows without a target or without forecasts."""
    return pd.read_parquet(config.PAIRS_PATH)


def load_training_data(models=("ecmwf",)):
    """Rows with an observed target and a forecast from every model in ``models``.

    ECMWF alone gives runs from 2024-03-14 (lead days 11-14 from 2024-07-16);
    adding ``"gfs"`` restricts the table to runs from 2026-04-02. GFS is missing
    before that date because the archive does not have it, so those rows must
    be left out, as here, never filled in.
    """
    pairs = load_pairs()
    keep = pairs[TARGET].notna()
    for model in models:
        keep &= pairs[f"{model}_t2m"].notna()
    return pairs[keep].reset_index(drop=True)


def load_forecast_features():
    """The 336 rows to predict: the Sep 16 12z run over the target window."""
    pairs = load_pairs()
    return pairs[pairs["run_time_utc"] == config.ISSUE_RUN].reset_index(drop=True)
