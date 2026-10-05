"""Train/validation splits by forecast run.

Rows are never split at random: the 336 rows of one run, and the rows of
neighbouring runs, describe the same weather. Validation runs always come after
all training runs, and the split is purged: every training target is observed
before the first validation target.
"""

import pandas as pd

from . import config


# Fold name -> (first, last) validation run date. Both sit in the same season
# as the target window. GFS exists only from 2026-04-02, so only the 2026 fold
# can validate models that need it.
FOLDS = {
    "autumn_2025": ("2025-08-15", "2025-10-15"),
    "late_summer_2026": ("2026-08-01", "2026-09-16"),
}


def split_by_run(data, val_start, val_end, purge=True):
    """Train on runs before ``val_start``; validate on runs in [val_start, val_end]."""
    val_start = pd.Timestamp(val_start).normalize()
    val_end = pd.Timestamp(val_end).normalize() + pd.Timedelta(days=1)
    runs = data["run_time_utc"]
    is_val = (runs >= val_start) & (runs < val_end)
    is_train = runs < val_start
    if purge and is_val.any():
        # A run from two weeks earlier forecasts hours that fall inside the
        # validation period. Keep only training targets that come before the
        # first validation target.
        first_val_target = runs[is_val].min() + pd.Timedelta(hours=config.FIRST_LEAD)
        is_train &= data["valid_time_utc"] < first_val_target
    return data[is_train], data[is_val]


def iter_folds(data, folds=None):
    """Yield (name, train, validation) for each fold, expanding-window style."""
    for name, (val_start, val_end) in (folds or FOLDS).items():
        train, val = split_by_run(data, val_start, val_end)
        if len(train) and len(val):
            yield name, train, val


def monthly_folds(first_month, last_month):
    """One fold per calendar month, e.g. ``monthly_folds("2025-07", "2026-08")``.

    Useful for tuning on more than the two default folds; each month is
    validated by a model trained on everything before it.
    """
    months = pd.period_range(first_month, last_month, freq="M")
    return {str(month): (month.start_time, month.end_time.normalize()) for month in months}
