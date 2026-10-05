"""RDU hourly observations (GHCNh routine reports) as a tidy table.

The raw files hold one routine report per hour, taken at minute 51. Each report
gets two labels:

- ``valid_time_utc``: the report time rounded to the next full hour (xx:51 ->
  xx+1:00). This is the key used to join NWP forecasts, which are on full hours.
- ``target_time_local``: the Eastern clock hour the report was taken in (00:51
  -> 00:00). "12am Sep 17" in the task is the report at 2026-09-17 00:51 local.

Every reading is kept as reported, with its GHCNh quality code (``*_qc``). The
temperature also keeps its source code, and ``temp_quality`` sorts it into:

- ``passed``      passed NCEI's quality control
- ``accepted``    usable but not a plain pass: checked against gross limits
                  only, flagged suspect and then accepted, edited by a
                  validator, or calculated
- ``unverified``  no quality code. This is every report since 2026-03-27
                  (source 413), which includes the target window
- ``rejected``    flagged suspect, erroneous or removed, or unverified and
                  implausible; ``obs_temp_c`` is then set to missing

The other variables are not screened: use their ``*_qc`` code before relying on
them.
"""

import numpy as np
import pandas as pd

from . import config


# Raw GHCNh column -> (column here, its quality-code column). Units: degC, degC, %, m/s, hPa.
OBS_COLUMNS = {
    "temperature": ("obs_temp_c", "temp_qc"),
    "dew_point_temperature": ("obs_dewpoint_c", "dewpoint_qc"),
    "relative_humidity": ("obs_rh_pct", "rh_qc"),
    "wind_speed": ("obs_wind_ms", "wind_qc"),
    "sea_level_pressure": ("obs_slp_hpa", "slp_qc"),
}

# What a quality code means depends on the source (GHCNh documentation, section
# VI, legacy codes). Any pair not listed here, including a blank code, counts as
# "unverified".
QUALITY_BY_SOURCE = {
    "343": {
        "1": "passed", "5": "passed",                          # passed all quality control checks
        "0": "accepted", "4": "accepted", "9": "accepted",     # passed the gross limits check only
        "A": "accepted",                                       # flagged suspect, accepted as good
        "P": "accepted", "U": "accepted", "I": "accepted",     # replaced, edited or inserted by a validator
        "M": "accepted", "R": "accepted",
        "2": "rejected", "6": "rejected",                      # suspect
        "3": "rejected", "7": "rejected",                      # erroneous
    },
    "223": {
        "1": "passed",                                         # good
        "4": "accepted",                                       # calculated
        "0": "unverified",                                     # not checked
        "2": "rejected", "3": "rejected", "5": "rejected",     # suspect, erroneous, removed
    },
}


def to_local(times_utc):
    """Naive UTC timestamps -> naive US Eastern wall-clock timestamps."""
    index = pd.DatetimeIndex(times_utc).tz_localize("UTC").tz_convert(config.LOCAL_TZ)
    return index.tz_localize(None)


def temperature_quality(source, code):
    """Quality class of each temperature reading from its source and quality code."""
    classes = [QUALITY_BY_SOURCE.get(s, {}).get(c, "unverified") for s, c in zip(source, code)]
    return pd.Series(classes, index=source.index)


def implausible(obs):
    """Readings outside -35..50 degC, or over 8 degC away from both neighbouring hours in the same direction.

    A thunderstorm can drop the temperature 10 degC in an hour, but it does not
    bounce back the next hour; a spike that does is an instrument or coding error.
    """
    temp = temperature_series(obs)
    from_before, from_after = temp - temp.shift(1), temp - temp.shift(-1)
    spike = (from_before.abs() > 8) & (from_after.abs() > 8) & (np.sign(from_before) == np.sign(from_after))
    bad = (temp < -35) | (temp > 50) | spike
    return bad.reindex(obs["valid_time_utc"]).to_numpy()


def read_ghcnh(raw_dir):
    """Read every yearly CSV under ``raw_dir`` into one tidy hourly table."""
    files = sorted(raw_dir.glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(f"no GHCNh files under {raw_dir}")
    wanted = {"DATE", "temperature_Source_Code"}
    for column in OBS_COLUMNS:
        wanted |= {column, f"{column}_Quality_Code"}
    # A column that is empty for a whole download is absent from its files.
    raw = pd.concat(
        (pd.read_csv(path, usecols=lambda column: column in wanted, dtype=str) for path in files),
        ignore_index=True,
    ).reindex(columns=sorted(wanted))

    obs = pd.DataFrame({"obs_time_utc": pd.to_datetime(raw["DATE"])})
    obs["valid_time_utc"] = obs["obs_time_utc"].dt.ceil("h")
    obs["target_time_local"] = to_local(obs["obs_time_utc"].dt.floor("h"))
    for column, (name, code) in OBS_COLUMNS.items():
        obs[name] = pd.to_numeric(raw[column])
        obs[code] = raw[f"{column}_Quality_Code"].fillna("")
    obs["temp_source"] = raw["temperature_Source_Code"].fillna("")
    obs = obs.dropna(subset=["obs_temp_c"])
    obs = obs.sort_values("obs_time_utc").drop_duplicates("valid_time_utc").reset_index(drop=True)

    quality = temperature_quality(obs["temp_source"], obs["temp_qc"])
    quality[(quality == "unverified") & implausible(obs)] = "rejected"
    obs.insert(obs.columns.get_loc("temp_qc") + 1, "temp_quality", quality)
    obs.loc[quality == "rejected", "obs_temp_c"] = np.nan
    return obs


def build_obs():
    """Observations usable for training: everything reported before the cutoff."""
    obs = read_ghcnh(config.RAW_OBS_DIR)
    if obs["obs_time_utc"].max() >= config.CUTOFF_UTC:
        raise ValueError("data/raw/ghcnh_rdu contains observations from after the cutoff")
    return obs


def load_obs():
    return pd.read_parquet(config.OBS_PATH)


def load_holdout():
    """Observed temperatures in the target window. For final scoring only.

    These were measured after the cutoff, so they must never reach training,
    feature building, or model selection.
    """
    obs = read_ghcnh(config.RAW_HOLDOUT_DIR)
    in_window = obs["target_time_local"].between(config.TARGET_START_LOCAL, config.TARGET_END_LOCAL)
    columns = ["target_time_local", "valid_time_utc", "obs_temp_c", "temp_quality"]
    return obs.loc[in_window, columns].reset_index(drop=True)


def temperature_series(obs):
    """Temperature on a gap-free hourly index (missing hours are NaN)."""
    series = obs.set_index("valid_time_utc")["obs_temp_c"]
    return series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="h"))
