"""Checks on the built tables: task geometry, time alignment, quality classes, and no use of post-cutoff data."""

import numpy as np
import pandas as pd
import pytest

from weather_modeling import config
from weather_modeling.climatology import load_climatology
from weather_modeling.dataset import (
    TARGET,
    TARGET_INFO,
    build_pairs,
    load_forecast_features,
    load_pairs,
    load_training_data,
)
from weather_modeling.nwp import MODELS, load_nwp
from weather_modeling.obs import load_obs, temperature_quality
from weather_modeling.station_break import before_break, estimate_offset, shift_before_break


pytestmark = pytest.mark.skipif(not config.PAIRS_PATH.exists(), reason="run scripts/build_dataset.py first")


@pytest.fixture(scope="module")
def pairs():
    return load_pairs()


@pytest.fixture(scope="module")
def forecasts():
    return {model: load_nwp(model) for model in MODELS}


def test_every_run_is_laid_out_like_the_task(pairs):
    assert (pairs["run_time_utc"].dt.hour == config.RUN_HOUR_UTC).all()
    assert (pairs.groupby("run_time_utc").size() == 336).all()
    assert pairs["lead_hour"].between(config.FIRST_LEAD, config.LAST_LEAD).all()
    assert (pairs.groupby(["run_time_utc", "lead_day"]).size() == 24).all()
    assert set(pairs["lead_day"]) == set(range(1, 15))


def test_forecast_rows_cover_exactly_the_target_window():
    rows = load_forecast_features()
    expected = pd.date_range(config.TARGET_START_LOCAL, config.TARGET_END_LOCAL, freq="h")
    assert list(rows["target_time_local"]) == list(expected)
    assert rows[TARGET].isna().all()
    assert rows[["ecmwf_t2m", "gfs_t2m", "clim_temp_c", "persist_temp_c"]].notna().all().all()


def test_nothing_from_after_the_cutoff_is_used(pairs):
    assert load_obs()["obs_time_utc"].max() < config.CUTOFF_UTC
    assert pairs["run_time_utc"].max() == config.ISSUE_RUN
    for model in MODELS:
        assert load_nwp(model)["run_time_utc"].max() <= config.ISSUE_RUN
    # A target exists only where the report was taken before the cutoff.
    assert pairs.loc[pairs[TARGET].notna(), "valid_time_utc"].max() <= config.CUTOFF_UTC


def test_no_duplicate_keys(pairs, forecasts):
    assert not load_obs()["valid_time_utc"].duplicated().any()
    assert not pairs.duplicated(["run_time_utc", "valid_time_utc"]).any()
    for frame in forecasts.values():
        assert not frame.duplicated(["run_time_utc", "valid_time_utc"]).any()


def test_target_is_the_report_taken_nine_minutes_before_valid_time(pairs):
    obs = load_obs().set_index("valid_time_utc")
    sample = pairs.dropna(subset=[TARGET]).sample(500, random_state=0)
    reports = obs.loc[sample["valid_time_utc"]]
    assert np.allclose(sample[TARGET].to_numpy(), reports["obs_temp_c"].to_numpy())
    assert (sample["valid_time_utc"].to_numpy() - reports["obs_time_utc"].to_numpy() == np.timedelta64(9, "m")).all()
    assert (sample["target_time_local"].to_numpy() == reports["target_time_local"].to_numpy()).all()


@pytest.mark.parametrize("run", ["2024-12-25 12:00", "2025-09-16 12:00", "2026-07-04 12:00"])
def test_features_survive_deleting_everything_after_the_cutoff(pairs, forecasts, run):
    """Rebuild one run from observations truncated at its cutoff: features must not move."""
    run = pd.Timestamp(run)
    obs = load_obs()
    truncated = obs[obs["valid_time_utc"] <= run + pd.Timedelta(hours=config.OBS_KNOWN_LEAD)]
    one_run = {model: frame[frame["run_time_utc"] == run] for model, frame in forecasts.items()}
    rebuilt = build_pairs(truncated, load_climatology(), one_run, runs=[run])

    stored = pairs[pairs["run_time_utc"] == run].reset_index(drop=True)
    features = [column for column in pairs.columns if column not in (TARGET, *TARGET_INFO)]
    pd.testing.assert_frame_equal(rebuilt[features], stored[features], check_dtype=False)
    assert rebuilt[TARGET].isna().all()


def test_a_report_belongs_to_its_clock_hour_and_the_first_target_is_after_the_cutoff():
    assert config.FIRST_LEAD == config.OBS_KNOWN_LEAD + 1 == 17
    first = load_forecast_features().iloc[0]
    # Label Sep 17 00:00 stands for the report taken at 00:51 that day, paired
    # with the 01:00 forecast (05:00 UTC) of the run started 17 hours earlier.
    assert first["target_time_local"] == pd.Timestamp("2026-09-17 00:00")
    assert first["valid_time_utc"] == pd.Timestamp("2026-09-17 05:00") > config.CUTOFF_UTC
    assert first["run_time_utc"] == config.ISSUE_RUN and pd.isna(first[TARGET])
    # The last observation available as an input was taken at 23:51 on Sep 16.
    last = load_obs().iloc[-1]
    assert last["target_time_local"] == pd.Timestamp("2026-09-16 23:00")
    assert last["obs_time_utc"] == pd.Timestamp("2026-09-17 03:51") < config.CUTOFF_UTC


def test_quality_code_meaning_depends_on_the_source():
    source = pd.Series(["343", "343", "343", "343", "223", "223", "223", "413", "999"])
    code = pd.Series(["5", "A", "7", "", "1", "4", "5", "", "1"])
    expected = ["passed", "accepted", "rejected", "unverified", "passed", "accepted", "rejected", "unverified", "unverified"]
    assert list(temperature_quality(source, code)) == expected


def test_every_temperature_has_a_quality_class_and_rejected_ones_are_blanked(pairs):
    obs = load_obs()
    assert set(obs["temp_quality"]) <= {"passed", "accepted", "unverified", "rejected"}
    rejected = obs["temp_quality"] == "rejected"
    assert rejected.any() and obs.loc[rejected, "obs_temp_c"].isna().all()
    assert obs.loc[~rejected, "obs_temp_c"].notna().all()
    # A rejected report is never a target; a known target always has a class.
    known = pairs[TARGET].notna()
    assert set(pairs.loc[known, "target_quality"]) <= {"passed", "accepted", "unverified"}
    # Recent reports, which include the validation targets of the 2026 fold, are not yet checked by NOAA.
    assert (obs.loc[obs["valid_time_utc"] >= "2026-04-01", "temp_quality"] == "unverified").all()


def test_readings_are_used_as_reported_and_the_break_is_only_flagged(pairs):
    """No shift in the stored tables: stored temperatures equal the raw files, on both sides of the break."""
    raw = pd.read_csv(
        next((config.RAW_OBS_DIR / "2025").glob("*.csv")), usecols=["DATE", "temperature"], parse_dates=["DATE"]
    ).set_index("DATE")["temperature"]
    stored = load_obs().set_index("obs_time_utc")["obs_temp_c"].reindex(raw.index).dropna()
    assert np.allclose(stored.to_numpy(), raw.reindex(stored.index).to_numpy())
    assert pairs["target_before_break"].any()
    assert not load_forecast_features()["target_before_break"].any()
    assert not before_break(pd.Series(pd.to_datetime([f"{config.CLIM_LAST_YEAR}-12-31", "2025-07-22"]))).any()


def test_break_offset_uses_only_data_from_before_as_of(forecasts):
    obs, ecmwf = load_obs(), forecasts["ecmwf"]
    as_of = pd.Timestamp("2025-08-16 04:00")
    offset, days = estimate_offset(obs, ecmwf, as_of)
    truncated = estimate_offset(obs[obs["valid_time_utc"] < as_of], ecmwf[ecmwf["valid_time_utc"] < as_of], as_of)
    assert (offset, days) == truncated and days == 25
    assert 0.3 < offset < 2.5
    # Too soon after the break to know about it, and never more than a year of data.
    assert estimate_offset(obs, ecmwf, pd.Timestamp("2025-07-30")) == (0.0, 8)
    assert estimate_offset(obs, ecmwf, config.CUTOFF_UTC)[1] == 365

    shifted = shift_before_break(obs, 1.0)
    moved = (obs["obs_temp_c"] - shifted["obs_temp_c"]).fillna(0)
    assert (moved[before_break(obs["valid_time_utc"])].round(6).isin([0.0, 1.0])).all()
    assert (moved[~before_break(obs["valid_time_utc"])] == 0).all()


def test_runs_without_proof_of_timely_publication_are_left_out(pairs, forecasts):
    from weather_modeling.nwp import EXCLUDED_RUNS

    for model, runs in EXCLUDED_RUNS.items():
        excluded = pd.to_datetime(runs)
        assert not forecasts[model]["run_time_utc"].isin(excluded).any()
        assert load_nwp(model, include_excluded=True)["run_time_utc"].isin(excluded).any()
        assert pairs.loc[pairs["run_time_utc"].isin(excluded), f"{model}_t2m"].isna().all()


def test_training_data_has_a_target_and_the_requested_forecasts():
    data = load_training_data(models=("ecmwf", "gfs"))
    assert data[[TARGET, "ecmwf_t2m", "gfs_t2m"]].notna().all().all()
    assert data["run_time_utc"].min() >= pd.Timestamp(MODELS["gfs"][1])
