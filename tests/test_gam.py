"""Checks on the linear GAM's inputs (no post-cutoff data, complete final rows) and a smoke fit."""

import numpy as np
import pandas as pd
import pytest

from weather_modeling import config
from weather_modeling.climatology import load_climatology
from weather_modeling.dataset import TARGET, load_pairs
from weather_modeling.gam import (
    GAM_FEATURES,
    LR_FEATURES,
    add_mos_features,
    build_mos_table,
    fit_lr,
    forecast_rows,
    predict,
    recency_weights,
    trainable,
)
from weather_modeling.nwp import load_nwp
from weather_modeling.obs import load_obs, temperature_series


pytestmark = pytest.mark.skipif(not config.PAIRS_PATH.exists(), reason="run scripts/build_dataset.py first")


@pytest.fixture(scope="module")
def table():
    return build_mos_table()


@pytest.fixture(scope="module")
def ecmwf():
    return load_nwp("ecmwf")


@pytest.mark.parametrize("run", ["2024-12-25 12:00", "2025-09-16 12:00", "2026-07-04 12:00", "2026-09-16 12:00"])
def test_mos_features_survive_deleting_everything_after_the_cutoff(table, ecmwf, run):
    """Rebuild one run from observations up to its cutoff and runs up to itself: nothing may move."""
    run = pd.Timestamp(run)
    obs = load_obs()
    truncated = obs[obs["valid_time_utc"] <= run + pd.Timedelta(hours=config.OBS_KNOWN_LEAD)]
    pairs = load_pairs()
    one_run = pairs[pairs["run_time_utc"] == run].reset_index(drop=True)
    rebuilt = add_mos_features(one_run, temperature_series(truncated), load_climatology(),
                               ecmwf[ecmwf["run_time_utc"] <= run])
    stored = table[table["run_time_utc"] == run].reset_index(drop=True)
    columns = sorted(set(LR_FEATURES) | set(GAM_FEATURES) | {"ecmwf_lagged_mean"})
    pd.testing.assert_frame_equal(rebuilt[columns], stored[columns], check_dtype=False)


def test_final_rows_have_every_input(table):
    final = forecast_rows(table)
    assert len(final) == 336 and final["run_time_utc"].eq(config.ISSUE_RUN).all()
    assert final[[*LR_FEATURES, *GAM_FEATURES, "clim_temp_c"]].notna().all().all()
    assert final[TARGET].isna().all()


def test_trainable_rows_match_the_shared_training_table(table):
    data = trainable(table)
    assert len(data) == 286_693
    assert data[[TARGET, *LR_FEATURES, *GAM_FEATURES]].notna().all().all()
    assert np.allclose(data["target_anomaly"], data[TARGET] - data["clim_temp_c"])


def test_recency_weights_halve_every_half_life():
    runs = pd.Series(pd.to_datetime(["2026-09-16", "2026-03-20", "2025-09-21"]))
    assert np.allclose(recency_weights(runs, half_life_days=180), [1.0, 0.5, 0.25])


def test_lr_port_predicts_climatology_plus_anomaly(table):
    data = trainable(table)
    models = fit_lr(data[data["run_time_utc"] < "2026-01-01"])
    final = forecast_rows(table)
    predicted = predict(models, final, LR_FEATURES)
    assert np.isfinite(predicted).all() and np.abs(predicted - final["clim_temp_c"]).max() < 15


def test_linear_gam_fits_and_predicts_every_final_hour(table):
    pytest.importorskip("pygam")
    from weather_modeling.gam import fit_gam

    data = trainable(table)
    recent = data[data["run_time_utc"] >= "2026-03-01"]
    models = fit_gam(recent, lam_grid=np.array([1.0, 100.0]))
    assert set(models) == set(range(1, 15))
    predicted = predict(models, forecast_rows(table), GAM_FEATURES)
    assert np.isfinite(predicted).all()
