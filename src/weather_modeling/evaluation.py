"""Shared evaluation harness: baselines, cross-validation, lead-day plots, final scoring.

Typical use::

    data = load_training_data(models=("ecmwf",))
    scored = cross_validate({"my_model": fit_predict}, data)
    table = score_folds(scored, [*BASELINES, "my_model"])

Baselines: ``persistence``, ``climatology``, ``station_lr`` (a regression on
station data only), ``raw_ecmwf`` and ``raw_gfs``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression

from . import config
from .climatology import load_climatology
from .dataset import TARGET, build_pairs, load_forecast_features, load_pairs
from .metrics import score_by
from .nwp import MODELS, load_nwp
from .obs import load_holdout, load_obs
from .splits import FOLDS, iter_folds, split_by_run
from .station_break import estimate_offset, shift_before_break


BASELINES = ["persistence", "climatology", "station_lr", "raw_ecmwf", "raw_gfs"]

# Baseline name -> the column of the pairs table that already is that forecast.
BASELINE_COLUMNS = {
    "persistence": "persist_temp_c",
    "climatology": "clim_temp_c",
    "raw_ecmwf": "ecmwf_t2m",
    "raw_gfs": "gfs_t2m",
}
STATION_LR_FEATURES = ["clim_temp_c", "anom_24h"]

# A treatment in break_sensitivity is skipped if it leaves fewer training runs than this.
MIN_TRAIN_RUNS = 60

# Colour follows the series, not its position, so a baseline keeps its colour
# in every figure. Models take the remaining slots in the order given.
BASELINE_COLORS = {
    "climatology": "#2a78d6",
    "persistence": "#eb6834",
    "raw_ecmwf": "#1baf7a",
    "raw_gfs": "#eda100",
    "station_lr": "#e87ba4",
}
MODEL_COLORS = ["#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def station_lr(train, val):
    """Station-only linear regression: climatology + last-24h anomaly, one fit per lead day.

    The best forecast we can make without any NWP model, so the gap between it
    and a model shows what the NWP forecasts add. A missing anomaly counts as 0.
    """
    predicted = pd.Series(np.nan, index=val.index)
    for lead_day, rows in val.groupby("lead_day"):
        fit = train[(train["lead_day"] == lead_day) & train[TARGET].notna()]
        model = LinearRegression().fit(fit[STATION_LR_FEATURES].fillna(0), fit[TARGET])
        predicted.loc[rows.index] = model.predict(rows[STATION_LR_FEATURES].fillna(0))
    return predicted.to_numpy()


def add_baselines(data, train):
    """Add one column per baseline to ``data``; ``train`` is what station_lr is fitted on."""
    columns = {name: data[column] for name, column in BASELINE_COLUMNS.items()}
    return data.assign(**columns, station_lr=station_lr(train, data))


def cross_validate(models, data, folds=None):
    """Out-of-fold predictions for every validation row of every fold.

    ``models`` maps a name to ``fit_predict(train, val_features)``, which must
    return one prediction per row of ``val_features``. The validation frame is
    passed without the target column. The result holds the validation rows with
    a ``fold`` column, the baselines, and one column per model.
    """
    parts = []
    for fold, train, val in iter_folds(data, folds):
        scored = add_baselines(val, train).assign(fold=fold)
        for name, fit_predict in models.items():
            predicted = np.asarray(fit_predict(train, val.drop(columns=TARGET)), dtype=float)
            if predicted.shape != (len(val),):
                raise ValueError(f"{name}: expected {len(val)} predictions, got shape {predicted.shape}")
            scored[name] = predicted
        parts.append(scored)
    return pd.concat(parts, ignore_index=True)


def score_folds(scored, predictions, metric="mae", by="lead_day", fahrenheit=False, overall="equal"):
    """``score_by`` for each fold, stacked into one table indexed by (fold, group)."""
    tables = {
        fold: score_by(part, predictions, metric=metric, by=by, fahrenheit=fahrenheit, overall=overall)
        for fold, part in scored.groupby("fold", sort=False)
    }
    return pd.concat(tables, names=["fold", by])


def break_sensitivity(models=None, folds=None, require=("ecmwf",)):
    """How much the July 2025 station break matters: three treatments scored on each fold.

    - ``as_reported``: every reading as reported (the primary analysis).
    - ``shifted``: readings before the break lowered by an offset estimated only
      from data available at the fold's first cutoff.
    - ``dropped``: training targets from before the break left out.

    Validation targets are the same in all three as long as the fold starts
    after the break. Returns one row per (fold, treatment): the offset, the days
    of post-break data behind it, the number of training runs, and the 14-day
    MAE of every baseline and model. A treatment with fewer than
    ``MIN_TRAIN_RUNS`` training runs is left empty.
    """
    models = models or {}
    names = [*BASELINES, *models]
    obs, clim = load_obs(), load_climatology()
    forecasts = {model: load_nwp(model) for model in MODELS}

    def trainable(pairs):
        keep = pairs[TARGET].notna()
        for model in require:
            keep &= pairs[f"{model}_t2m"].notna()
        return pairs[keep].reset_index(drop=True)

    reported = trainable(load_pairs())
    rows = []
    for fold, (val_start, val_end) in (folds or FOLDS).items():
        train, val = split_by_run(reported, val_start, val_end)
        as_of = val["run_time_utc"].min() + pd.Timedelta(hours=config.OBS_KNOWN_LEAD)
        offset, days = estimate_offset(obs, forecasts["ecmwf"], as_of)
        shifted = trainable(build_pairs(shift_before_break(obs, offset), clim, forecasts))
        treatments = {
            "as_reported": (train, val),
            "shifted": split_by_run(shifted, val_start, val_end),
            "dropped": (train[~train["target_before_break"]], val),
        }
        for treatment, (fit, held_out) in treatments.items():
            row = {"fold": fold, "treatment": treatment, "train_runs": fit["run_time_utc"].nunique()}
            if treatment == "shifted":
                row.update(offset_c=offset, post_break_days=days)
            if row["train_runs"] >= MIN_TRAIN_RUNS:
                scored = add_baselines(held_out, fit)
                for name, fit_predict in models.items():
                    scored[name] = np.asarray(fit_predict(fit, held_out.drop(columns=TARGET)), dtype=float)
                row.update(score_by(scored, names).loc["all", names])
            rows.append(row)
    columns = ["offset_c", "post_break_days", "train_runs", *names]
    return pd.DataFrame(rows).set_index(["fold", "treatment"]).reindex(columns=columns)


def chart_style():
    """Matplotlib settings shared by the project's figures: quiet axes and grid, title top left."""
    # Use the system sans where there is one; asking for a missing family makes matplotlib warn.
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ("Segoe UI", "Helvetica Neue", "Arial") if name in installed), "DejaVu Sans")
    return {
        "font.family": family,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 200,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": AXIS,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.titleweight": "semibold", "axes.titlesize": 12, "axes.titlepad": 34,
        "legend.frameon": False, "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 6.5, "lines.markeredgewidth": 1.5, "lines.markeredgecolor": SURFACE,
    }


def legend_above(ax):
    """One-row legend between the title and the plot."""
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles), handlelength=1.4, columnspacing=1.4, borderaxespad=0.2)


def plot_by_lead_day(table, title, ylabel="MAE (°C)", path=None):
    """Line chart of a ``score_by`` table (one fold): one line per prediction column."""
    series = [column for column in table.columns if column != "n" and table[column].notna().any()]
    models = [name for name in series if name not in BASELINE_COLORS]
    if len(models) > len(MODEL_COLORS):
        raise ValueError(f"at most {len(MODEL_COLORS)} models per figure; plot fewer or use two figures")
    colors = {**BASELINE_COLORS, **dict(zip(models, MODEL_COLORS))}
    days = table.drop(index="all", errors="ignore")

    with plt.rc_context(chart_style()):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for name in series:
            ax.plot(days.index.astype(int), days[name], label=name, color=colors[name], marker="o")
        ax.set_ylim(bottom=0)
        ax.set_xticks(days.index.astype(int))
        ax.set_xlabel("Lead day")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        legend_above(ax)
        fig.tight_layout()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
    return fig


def check_submission(submission):
    """A submission is 336 hourly rows covering exactly the target window, with no gaps."""
    expected = pd.date_range(config.TARGET_START_LOCAL, config.TARGET_END_LOCAL, freq="h")
    times = pd.to_datetime(submission["target_time_local"])
    if len(times) != len(expected) or not (times.to_numpy() == expected.to_numpy()).all():
        raise ValueError(f"expected {len(expected)} hourly rows from {expected[0]} to {expected[-1]} (local)")
    if submission["pred_temp_c"].isna().any():
        raise ValueError("submission contains missing predictions")


def make_submission(predictions, name):
    """Write ``reports/predictions/<name>.csv`` from predictions for ``load_forecast_features()`` rows."""
    features = load_forecast_features()
    submission = pd.DataFrame(
        {"target_time_local": features["target_time_local"], "pred_temp_c": np.asarray(predictions, dtype=float)}
    )
    check_submission(submission)
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PREDICTIONS_DIR / f"{name}.csv"
    submission.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M")
    return path


def holdout_frame(submissions=None):
    """The target window with the observed temperature, baselines and submissions.

    Final scoring only: the observations here are from after the cutoff. Do not
    use this to choose features, hyperparameters or models.
    """
    frame = add_baselines(load_forecast_features().drop(columns=TARGET), load_pairs())
    truth = load_holdout().set_index("target_time_local")
    frame[TARGET] = frame["target_time_local"].map(truth[TARGET])
    frame["target_quality"] = frame["target_time_local"].map(truth["temp_quality"])
    for name, submission in (submissions or {}).items():
        check_submission(submission)
        frame[name] = submission["pred_temp_c"].to_numpy()
    return frame
