"""Checks on splits, metrics and the evaluation harness, using a small synthetic table."""

import numpy as np
import pandas as pd
import pytest

from weather_modeling import config
from weather_modeling.dataset import TARGET
from weather_modeling.evaluation import BASELINE_COLUMNS, BASELINES, check_submission, cross_validate, score_folds
from weather_modeling.metrics import bootstrap_mae_difference, score_by
from weather_modeling.splits import monthly_folds, split_by_run


@pytest.fixture
def toy():
    """90 daily runs with the real lead geometry and made-up temperatures."""
    runs = pd.date_range("2026-01-01 12:00", periods=90, freq="D")
    leads = np.arange(config.FIRST_LEAD, config.LAST_LEAD + 1)
    data = pd.MultiIndex.from_product([runs, leads], names=["run_time_utc", "lead_hour"]).to_frame(index=False)
    data["valid_time_utc"] = data["run_time_utc"] + pd.to_timedelta(data["lead_hour"], unit="h")
    data["lead_day"] = (data["lead_hour"] - config.FIRST_LEAD) // 24 + 1
    rng = np.random.default_rng(0)
    data[TARGET] = 10 + rng.normal(size=len(data))
    for column in [*BASELINE_COLUMNS.values(), "anom_24h"]:
        data[column] = data[TARGET] + rng.normal(size=len(data))
    return data


def test_validation_runs_come_after_training_runs(toy):
    train, val = split_by_run(toy, "2026-03-01", "2026-03-15")
    assert train["run_time_utc"].max() < val["run_time_utc"].min()
    assert val["run_time_utc"].dt.normalize().nunique() == 15


def test_purged_training_targets_all_come_before_the_first_validation_target(toy):
    purged, val = split_by_run(toy, "2026-03-01", "2026-03-15")
    unpurged, _ = split_by_run(toy, "2026-03-01", "2026-03-15", purge=False)
    assert purged["valid_time_utc"].max() < val["valid_time_utc"].min()
    assert unpurged["valid_time_utc"].max() > val["valid_time_utc"].min()
    assert len(purged) < len(unpurged)


def test_monthly_folds_cover_whole_months():
    folds = monthly_folds("2026-01", "2026-03")
    assert list(folds) == ["2026-01", "2026-02", "2026-03"]
    assert folds["2026-02"] == (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-28"))


def test_overall_score_weights_lead_days_equally():
    data = pd.DataFrame({"lead_day": [1, 1, 1, 2], TARGET: [0.0, 0.0, 0.0, 0.0], "model": [1.0, 1.0, 1.0, 5.0]})
    table = score_by(data, ["model"])
    assert table.loc[1, "model"] == 1.0 and table.loc[2, "model"] == 5.0
    assert table.loc["all", "model"] == 3.0  # not the pooled 2.0
    assert table.loc["all", "n"] == 4
    assert score_by(data, ["model"], fahrenheit=True).loc["all", "model"] == pytest.approx(5.4)
    assert score_by(data, ["model"], metric="rmse").loc["all", "model"] == pytest.approx(np.sqrt(13))
    assert score_by(data, ["model"], metric="bias").loc["all", "model"] == 3.0
    # Pooling the hours instead gives a different number from the same forecasts.
    assert score_by(data, ["model"], overall="pooled").loc["all", "model"] == 2.0
    assert score_by(data, ["model"], metric="rmse", overall="pooled").loc["all", "model"] == pytest.approx(np.sqrt(7))


def test_models_are_compared_on_the_same_rows():
    data = pd.DataFrame(
        {
            "lead_day": [1, 1, 1],
            TARGET: [0.0, 0.0, 0.0],
            "a": [1.0, 1.0, 9.0],
            "b": [2.0, 2.0, np.nan],
            "never_available": [np.nan] * 3,
        }
    )
    table = score_by(data, ["a", "b", "never_available"])
    assert table.loc[1, "a"] == 1.0  # the row where b is missing is dropped for a too
    assert table.loc[1, "n"] == 2
    assert np.isnan(table.loc[1, "never_available"])


def test_cross_validate_hides_the_target_and_returns_out_of_fold_predictions(toy):
    seen = []

    def fit_predict(train, val):
        assert TARGET in train.columns and TARGET not in val.columns
        seen.append((train["run_time_utc"].max(), val["run_time_utc"].min()))
        return np.full(len(val), train[TARGET].mean())

    folds = {"feb": ("2026-02-01", "2026-02-28"), "mar": ("2026-03-01", "2026-03-31")}
    scored = cross_validate({"mean_model": fit_predict}, toy, folds=folds)
    assert all(last_train < first_val for last_train, first_val in seen)
    assert set(scored["fold"]) == {"feb", "mar"}
    table = score_folds(scored, [*BASELINES, "mean_model"])
    assert table.loc[("mar", "all"), "mean_model"] > 0
    assert table.loc[("feb", "all"), "n"] == 28 * 336
    assert table.loc[("mar", "all"), BASELINES].notna().all()


def test_bootstrap_interval_detects_a_clearly_better_model(toy):
    rng = np.random.default_rng(1)
    toy = toy.assign(
        good=toy[TARGET] + rng.normal(scale=0.5, size=len(toy)),
        bad=toy[TARGET] + rng.normal(scale=3.0, size=len(toy)),
        twin=toy[TARGET] + rng.normal(scale=0.5, size=len(toy)),
    )
    better = bootstrap_mae_difference(toy, "good", "bad", n_boot=200)
    assert better["low"] < better["difference"] < better["high"] < 0
    same = bootstrap_mae_difference(toy, "good", "twin", n_boot=200)
    assert same["low"] < 0 < same["high"]


def test_submission_must_cover_exactly_the_target_window():
    hours = pd.date_range(config.TARGET_START_LOCAL, config.TARGET_END_LOCAL, freq="h")
    good = pd.DataFrame({"target_time_local": hours, "pred_temp_c": 20.0})
    check_submission(good)
    with pytest.raises(ValueError):
        check_submission(good.iloc[1:])
    with pytest.raises(ValueError):
        check_submission(good.assign(pred_temp_c=np.nan))
