"""Hourly temperature climatology for RDU: the multi-year mean by day of year and UTC hour."""

import numpy as np
import pandas as pd

from . import config


def day_of_year_365(times):
    """Day of year on a 365-day calendar (Feb 29 shares a slot with Feb 28)."""
    index = pd.DatetimeIndex(times)
    doy = index.dayofyear.to_numpy()
    return np.where(index.is_leap_year & (doy > 59), doy - 1, doy)


def fit_climatology(obs, first_year=config.CLIM_FIRST_YEAR, last_year=config.CLIM_LAST_YEAR, half_window=10):
    """Mean temperature per (day of year, UTC hour), smoothed over +-``half_window`` days.

    UTC hours keep the same solar time all year, so daylight saving does not
    smear the diurnal cycle. Returns a 365 x 24 table (index ``doy`` 1..365,
    columns ``hour_utc`` 0..23).
    """
    years = obs["valid_time_utc"].dt.year
    sample = obs[(years >= first_year) & (years <= last_year)]
    table = (
        sample.assign(doy=day_of_year_365(sample["valid_time_utc"]), hour_utc=sample["valid_time_utc"].dt.hour)
        .groupby(["doy", "hour_utc"])["obs_temp_c"]
        .mean()
        .unstack("hour_utc")
    )
    # Wrap around the year end so early January and late December smooth each other.
    wrapped = pd.concat([table.iloc[-half_window:], table, table.iloc[:half_window]])
    smooth = wrapped.rolling(2 * half_window + 1, center=True).mean().iloc[half_window:-half_window]
    smooth.index.name, smooth.columns.name = "doy", "hour_utc"
    return smooth


def climatology_at(clim, valid_time_utc):
    """Climatological temperature at each UTC valid time."""
    index = pd.DatetimeIndex(valid_time_utc)
    return clim.to_numpy()[day_of_year_365(index) - 1, index.hour]


def save_climatology(clim):
    clim.rename(columns=str).to_parquet(config.CLIMATOLOGY_PATH)


def load_climatology():
    clim = pd.read_parquet(config.CLIMATOLOGY_PATH)
    clim.columns = clim.columns.astype(int)
    clim.columns.name = "hour_utc"
    return clim
