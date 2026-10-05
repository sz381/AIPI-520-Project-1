"""Error metrics, by lead day or any other grouping. Errors are in degC unless ``fahrenheit=True``."""

import numpy as np
import pandas as pd

from .dataset import TARGET


def mae(observed, predicted):
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(observed))))


def rmse(observed, predicted):
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(observed)) ** 2)))


def bias(observed, predicted):
    """Mean (predicted - observed): positive means the prediction runs warm."""
    return float(np.mean(np.asarray(predicted) - np.asarray(observed)))


def score_by(data, predictions, metric="mae", by="lead_day", target=TARGET, fahrenheit=False, overall="equal"):
    """Table of ``metric`` per group, one column per prediction column.

    All predictions are scored on the same rows: those where the target and
    every prediction are present. A prediction that is entirely missing (e.g.
    GFS before 2026-04) is reported as NaN instead of emptying the comparison.

    The ``all`` row weights every group equally by default. With
    ``by="lead_day"`` that matches the real task, where each lead day contributes
    24 hours, even when late runs have fewer verified long-lead rows.
    ``overall="pooled"`` weights every row equally instead. The two agree only
    when all groups are the same size. ``n`` is the row count.
    """
    if overall not in ("equal", "pooled"):
        raise ValueError(f"unknown overall {overall!r}; use 'equal' or 'pooled'")
    usable = [column for column in predictions if data[column].notna().any()]
    rows = data.dropna(subset=[target, *usable])
    error = rows[usable].sub(rows[target], axis=0) * (1.8 if fahrenheit else 1.0)
    groups = rows[by]

    pooled = overall == "pooled"
    if metric == "mae":
        table = error.abs().groupby(groups).mean()
        total = error.abs().mean() if pooled else table.mean()
    elif metric == "rmse":
        mean_square = (error**2).groupby(groups).mean()
        table = np.sqrt(mean_square)
        total = np.sqrt((error**2).mean() if pooled else mean_square.mean())
    elif metric == "bias":
        table = error.groupby(groups).mean()
        total = error.mean() if pooled else table.mean()
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'mae', 'rmse' or 'bias'")

    table.loc["all"] = total
    table = table.reindex(columns=list(predictions))
    table["n"] = groups.value_counts().reindex(table.index).fillna(len(rows)).astype(int)
    return table


def bootstrap_mae_difference(data, pred_a, pred_b, target=TARGET, block_days=14, n_boot=2000, seed=0):
    """MAE(pred_a) - MAE(pred_b) with a 95% interval; negative favours ``pred_a``.

    Resamples blocks of consecutive runs rather than single rows. Rows of one
    run, and of runs up to 14 days apart, are scored against the same hours and
    share their errors; treating them as independent would make the interval far
    too narrow. Blocks of 14 days cover that overlap, but a fold then holds only
    three to five blocks, so read the interval as a rough guide. MAE here is
    pooled over rows (not equal-weighted by lead day).
    """
    rows = data.dropna(subset=[target, pred_a, pred_b])
    gap = (rows[pred_a] - rows[target]).abs() - (rows[pred_b] - rows[target]).abs()
    block = (rows["run_time_utc"] - rows["run_time_utc"].min()).dt.days // block_days
    per_block = gap.groupby(block).agg(["sum", "count"]).to_numpy()

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_block), size=(n_boot, len(per_block)))
    resampled = per_block[draws].sum(axis=1)
    samples = resampled[:, 0] / resampled[:, 1]
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "difference": float(gap.mean()),
        "low": float(low),
        "high": float(high),
        "n_blocks": len(per_block),
    }
