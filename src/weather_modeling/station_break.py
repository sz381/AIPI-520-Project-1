"""The July 2025 break in RDU's temperature record.

Around 2025-07-22 the station's readings drop by roughly 0.5-1.0 degC relative
to ECMWF forecasts, the ERA5 reanalysis and neighbouring stations, after sitting
higher since early 2024 (scripts/check_station_break.py). That shows the two
periods were measured differently. It does not show which one is right, and the
cause is not confirmed.

The primary analysis therefore uses every reading as reported. This module
supports two sensitivity treatments (see ``evaluation.break_sensitivity``):

- shift the earlier readings onto the current scale by an offset estimated only
  from data available at the time (``estimate_offset``);
- leave targets from before the break out of training (``before_break``).
"""

import pandas as pd

from . import config


# Below this many days of post-break data the break is treated as not yet knowable.
MIN_DAYS = 14


def before_break(valid_time_utc):
    """True for the period whose readings sit higher than the current ones."""
    return (valid_time_utc >= config.STATION_BREAK_START_UTC) & (valid_time_utc < config.STATION_BREAK_UTC)


def estimate_offset(obs, ecmwf, as_of):
    """How much higher the pre-break readings sit, using only data from before ``as_of``.

    Compares observed minus ECMWF (12-35h ahead) over the days after the break
    with the same number of days before it. It takes every post-break day
    available at ``as_of``, capped at one year: a full year against a full year
    cancels the seasonal cycle of the forecast error, a shorter window does not.

    Returns ``(offset in degC, days used)``; the offset is 0 when fewer than
    ``MIN_DAYS`` post-break days exist.
    """
    days = min((as_of - config.STATION_BREAK_UTC).days, 365)
    if days < MIN_DAYS:
        return 0.0, max(days, 0)
    reference = ecmwf[ecmwf["lead_hour"].between(12, 35)].set_index("valid_time_utc")["t2m"]
    observed = obs.set_index("valid_time_utc")["obs_temp_c"]
    gap = (observed.reindex(reference.index) - reference).dropna()
    gap = gap[gap.index < as_of]

    window = pd.Timedelta(days=days)
    after = gap[(gap.index >= config.STATION_BREAK_UTC) & (gap.index < config.STATION_BREAK_UTC + window)]
    before = gap[(gap.index >= config.STATION_BREAK_UTC - window) & (gap.index < config.STATION_BREAK_UTC)]
    return float(before.mean() - after.mean()), days


def shift_before_break(obs, offset):
    """A copy of the observations with the pre-break temperatures lowered by ``offset``."""
    shifted = obs.copy()
    shifted.loc[before_break(shifted["valid_time_utc"]), "obs_temp_c"] -= offset
    return shifted
