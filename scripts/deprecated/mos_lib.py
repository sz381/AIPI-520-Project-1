"""Shared data loading, sample construction, folds and metrics for the NWP (MOS) experiments.

Leakage rules enforced here:
- A forecast for run day d uses the 12z NWP run of day d and observations up to 23:51 local on day d
  (the "origin" row, 11pm local), exactly like the real forecast issued from the Sep 16, 2026 12z run.
- Climatology and the station state features come from 03 (past-only climatology, trailing windows).
- The recent NWP error only uses observed hours between the run time and the origin.
- Folds are purged: training rows must have target times before the validation period starts.
"""

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NY = ZoneInfo("America/New_York")
FEATURES_FILE = PROJECT_ROOT / "data" / "interim" / "rdu_hourly_features.csv"
NWP_DIR = PROJECT_ROOT / "data" / "external" / "nwp"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"

HORIZON = 336               # 12am day d+1 ... 11pm day d+14 (local)
RUN_HOUR_UTC = 12           # NWP run used for a forecast issued on the evening of day d
NWP_VALID_OFFSET_H = 1      # obs at hh:51 (labelled hh, floor convention) <-> model value valid at hh+1:00 UTC
RECENT_ERROR_HOURS = 6      # recent NWP error = mean(obs - NWP) over the last 6 observed hours before the origin
STATE_FEATURES = ["anom_now", "anom_24h", "anom_7d", "anom_30d",
                  "dewpoint_depression", "pressure_change_24h", "cloud_cover_octas"]
NWP_VARIABLES = ["temperature_2m", "dew_point_2m", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
                 "precipitation", "pressure_msl", "shortwave_radiation"]

FOLDS = {  # validation run days (inclusive); training rows must have target times before the start
    "fold1_2025": ("2025-08-15", "2025-10-15"),
    "fold2_2026": ("2026-08-01", "2026-09-16"),
}


def load_hourly():
    """Hourly table from 03: temperature, past-only climatology, calendar and station state features."""
    hourly = pd.read_csv(FEATURES_FILE, index_col="hour_utc", parse_dates=["hour_utc"])
    hourly["hour_local"] = pd.to_datetime(hourly["hour_local"], utc=True).dt.tz_convert(NY)
    assert hourly.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all(), "hourly grid must be continuous"
    return hourly


ECMWF_FILE = PROJECT_ROOT / "data" / "raw" / "ecmwf_ifs_runs_2024-03-14_to_2026-09-16.csv"  # from 01_data_sourcing_new


def load_nwp(model="ecmwf_ifs", run_hour=RUN_HOUR_UTC):
    """ECMWF runs initialised at `run_hour` UTC, from the file written by 01_data_sourcing_new."""
    assert model == "ecmwf_ifs"
    nwp = pd.read_csv(ECMWF_FILE)
    nwp = nwp[pd.to_datetime(nwp["run_utc"], utc=True).dt.hour.eq(run_hour)]
    nwp["run_utc"] = pd.to_datetime(nwp["run_utc"], utc=True, errors="coerce")
    nwp["valid_utc"] = pd.to_datetime(nwp["valid_utc"], utc=True, errors="coerce")
    nwp = nwp.dropna(subset=["run_utc", "valid_utc"])
    # keep the most complete copy when a run was fetched twice (e.g. an interrupted, half-written last line)
    nwp = nwp.assign(_filled=nwp[NWP_VARIABLES].notna().sum(axis=1)).sort_values("_filled", ascending=False)
    return nwp.drop_duplicates(["run_utc", "valid_utc"]).drop(columns="_filled").sort_values(["run_utc", "valid_utc"])


def historical_average(hourly, half_window_days=3):
    """Heuristic: mean of EARLIER years at the same local day-of-year and hour, smoothed over +-3 days."""
    local = hourly["hour_local"]
    frame = pd.DataFrame({"t": hourly["temperature"].to_numpy(), "year": local.dt.year.to_numpy(),
                          "doy": local.dt.dayofyear.to_numpy(), "hour": local.dt.hour.to_numpy()}, index=hourly.index)
    out = pd.Series(np.nan, index=hourly.index)
    for year in range(frame["year"].min() + 1, frame["year"].max() + 1):
        grid = frame[frame["year"] < year].groupby(["doy", "hour"])["t"].mean().unstack().reindex(range(1, 367))
        w = half_window_days
        smooth = pd.concat([grid.iloc[-w:], grid, grid.iloc[:w]]).rolling(2 * w + 1, center=True, min_periods=1).mean()
        smooth = smooth.iloc[w:-w].to_numpy()
        rows = frame["year"].eq(year).to_numpy()
        out[rows] = smooth[frame.loc[rows, "doy"].to_numpy() - 1, frame.loc[rows, "hour"].to_numpy()]
    return out


def build_samples(hourly, nwp, first_run_day="2024-03-14", last_run_day="2026-09-16"):
    """One row per (run day, target hour). Station state from the 11pm origin row, NWP from the 12z run."""
    local = hourly["hour_local"]
    run_days = pd.date_range(first_run_day, last_run_day, freq="D")
    origin_local = [pd.Timestamp(d.year, d.month, d.day, 23, tz=NY) for d in run_days]
    origin_pos = hourly.index.get_indexer(pd.DatetimeIndex(origin_local).tz_convert("UTC"))
    keep = origin_pos >= 0
    run_days, origin_pos = run_days[keep], origin_pos[keep]

    step = np.arange(1, HORIZON + 1)
    o = np.repeat(origin_pos, HORIZON)
    k = np.tile(step, len(origin_pos))
    t = o + k
    valid_rows = t < len(hourly)
    o, k, t = o[valid_rows], k[valid_rows], t[valid_rows]
    run_day = np.repeat(run_days.to_numpy(), HORIZON)[valid_rows]

    col = lambda name, pos: hourly[name].to_numpy()[pos]
    samples = pd.DataFrame({
        "run_day": pd.to_datetime(run_day),
        "run_utc": pd.to_datetime(run_day).tz_localize("UTC") + pd.Timedelta(hours=RUN_HOUR_UTC),
        "origin_utc": hourly.index[o],
        "target_utc": hourly.index[t],
        "step": k,                                  # hours after the origin (1 = 12am next day)
        "lead_day": (k - 1) // 24 + 1,
        "temperature": col("temperature", t),
        "climatology": col("climatology", t),
        "target_hour": local.dt.hour.to_numpy()[t],
        **{c: col(c, t) for c in ["hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2", "doy_sin1", "doy_cos1"]},
        **{c: col(c, o) for c in STATE_FEATURES},
        "persistence": col("temperature", t - 24 * np.ceil(k / 24).astype(int)),
    })
    samples["valid_utc"] = samples["target_utc"] + pd.Timedelta(hours=NWP_VALID_OFFSET_H)
    samples["lead_hours"] = (samples["valid_utc"] - samples["run_utc"]) // pd.Timedelta(hours=1)

    nwp_cols = nwp.rename(columns={v: f"nwp_{v}" for v in NWP_VARIABLES})
    samples = samples.merge(nwp_cols.drop(columns="lead_hours"), on=["run_utc", "valid_utc"], how="left")

    # Recent NWP error: obs minus forecast over the last RECENT_ERROR_HOURS observed hours up to the origin
    err_rows = []
    for back in range(RECENT_ERROR_HOURS):
        pos = origin_pos - back
        err_rows.append(pd.DataFrame({
            "run_utc": pd.to_datetime(run_days).tz_localize("UTC") + pd.Timedelta(hours=RUN_HOUR_UTC),
            "valid_utc": hourly.index[pos] + pd.Timedelta(hours=NWP_VALID_OFFSET_H),
            "obs": hourly["temperature"].to_numpy()[pos],
        }))
    err = pd.concat(err_rows).merge(nwp[["run_utc", "valid_utc", "temperature_2m"]], on=["run_utc", "valid_utc"])
    assert (err["valid_utc"] - pd.Timedelta(hours=NWP_VALID_OFFSET_H) > err["run_utc"]).all(), "error hours precede the run"
    err["e"] = err["obs"] - err["temperature_2m"]
    recent = err.groupby("run_utc")["e"].mean().rename("nwp_recent_error")
    samples = samples.merge(recent, left_on="run_utc", right_index=True, how="left")

    samples["nwp_anomaly"] = samples["nwp_temperature_2m"] - samples["climatology"]
    samples["hist_avg"] = historical_average(hourly).to_numpy()[hourly.index.get_indexer(samples["target_utc"])]
    assert (samples["origin_utc"] < samples["target_utc"]).all()
    return samples


def attach_lagged_run(samples, nwp_other, hours_earlier, name):
    """Temperature from an OLDER run at the same valid time (time-lagged ensemble member).

    hours_earlier=12 with the 00z file -> 00z run of day d; hours_earlier=24 with the 12z file -> 12z run of day d-1.
    Both are published hours before the 11pm origin, so they are available at forecast time.
    """
    other = nwp_other[["run_utc", "valid_utc", "temperature_2m"]].rename(columns={"temperature_2m": name})
    other = other.assign(run_utc=other["run_utc"] + pd.Timedelta(hours=hours_earlier))  # key it to the newer run
    return samples.merge(other, on=["run_utc", "valid_utc"], how="left")


def attach_rolling_bias(samples, hourly, nwp, window_days=30, leads=(17, 40)):
    """Mean (obs - NWP) over runs d-window .. d-2 at leads 17-40 h: all verified before the origin of day d."""
    verify = nwp[nwp["lead_hours"].between(*leads)][["run_utc", "valid_utc", "temperature_2m"]].copy()
    verify["obs"] = hourly["temperature"].reindex(verify["valid_utc"] - pd.Timedelta(hours=NWP_VALID_OFFSET_H)).to_numpy()
    daily = (verify["obs"] - verify["temperature_2m"]).groupby(verify["run_utc"]).mean()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    # run d-2 at lead 40 h verifies at d 04 UTC, i.e. before the 11pm-local origin of day d (03-04 UTC on d+1)
    rolling = daily.shift(2).rolling(window_days, min_periods=window_days // 2).mean().rename(f"nwp_bias_{window_days}d")
    return samples.merge(rolling, left_on="run_utc", right_index=True, how="left")


def fold_split(samples, fold):
    start, end = (pd.Timestamp(x) for x in FOLDS[fold])
    start_utc = pd.Timestamp(start.year, start.month, start.day, tz=NY).tz_convert("UTC")
    val = samples["run_day"].between(start, end) & samples["temperature"].notna()
    train = (samples["target_utc"] < start_utc) & samples["temperature"].notna()
    assert samples.loc[train, "target_utc"].max() < start_utc  # purge
    return samples[train], samples[val]


def rmse(e):
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e ** 2)))


def mae(e):
    return float(np.mean(np.abs(np.asarray(e, dtype=float))))


def score_table(predictions):
    """predictions: DataFrame with model, fold, lead_day, error."""
    overall = predictions.groupby(["model", "fold"])["error"].agg(RMSE=rmse, MAE=mae).unstack("fold")
    overall.columns = [f"{m}_{f}" for m, f in overall.columns]
    pooled = predictions.groupby("model")["error"].agg(RMSE_pooled=rmse, MAE_pooled=mae)
    return overall.join(pooled).sort_values("MAE_pooled")


def by_lead_day(predictions, metric=mae):
    return predictions.groupby(["model", "lead_day"])["error"].apply(metric).unstack("lead_day")
