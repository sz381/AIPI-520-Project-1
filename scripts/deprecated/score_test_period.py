"""Score every method on the test window (Sep 17-30, 2026) - run once, after the forecast was frozen.

All models are trained on rows with target times before 12am Sep 17 and predict from the Sep 16 12z run, exactly
as in make_final_forecast.py. The pre-selected model (M14) is checked against the frozen forecast file.
"""

import hashlib

import numpy as np
import pandas as pd

import mos_lib as lib
import run_mos_experiments as exp

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 60)
CUTOFF_UTC = pd.Timestamp("2026-09-17", tz=lib.NY).tz_convert("UTC")
OBS_FILE = lib.PROJECT_ROOT / "data" / "external" / "rdu_obs_2026-09-17_to_2026-09-30.csv"
FROZEN = lib.RESULTS_DIR / "final_forecast_M14.csv"
SELECTED = "M14 M11 with historical-average base"


def main():
    expected_hash = (lib.RESULTS_DIR / "final_forecast_M14.sha256").read_text().split()[0]
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == expected_hash, "frozen forecast file was modified"

    samples, _, _ = exp.prepare_samples(verbose=False)
    train = samples[(samples["target_utc"] < CUTOFF_UTC) & samples["temperature"].notna()]
    test = samples[samples["run_day"].eq(pd.Timestamp("2026-09-16"))].reset_index(drop=True)
    assert len(test) == lib.HORIZON and train["target_utc"].max() < CUTOFF_UTC

    obs = pd.read_csv(OBS_FILE, index_col="hour_utc", parse_dates=["hour_utc"])
    actual = obs["observed_deg_c"].reindex(test["target_utc"]).to_numpy()  # column written by notebook 05
    print(f"Test hours with an observation: {np.isfinite(actual).sum()} of {len(actual)}")

    fallback = exp.station_lr(train, test)
    rows, by_day = {}, {}
    for name, fit_predict in exp.MODELS.items():
        pred = fit_predict(train, test)
        pred = np.where(np.isnan(pred), fallback, pred)
        pred = np.where(np.isnan(pred), test["climatology"].to_numpy(), pred)
        if name == SELECTED:
            frozen = pd.read_csv(FROZEN)["temperature_forecast_c"].to_numpy()
            assert np.allclose(pred.round(2), frozen), "re-computed M14 differs from the frozen forecast"
        err = actual - pred
        ok = np.isfinite(err)
        rows[name] = {"MAE": lib.mae(err[ok]), "RMSE": lib.rmse(err[ok]), "bias": float(np.mean(err[ok]))}
        by_day[name] = pd.Series(err[ok]).groupby(test["lead_day"].to_numpy()[ok]).apply(lib.mae)

    table = pd.DataFrame(rows).T.round(3)
    days = pd.DataFrame(by_day).T.round(2)
    days.columns.name = "lead_day"
    table.to_csv(lib.RESULTS_DIR / "test_scores.csv")
    days.to_csv(lib.RESULTS_DIR / "test_mae_by_lead_day.csv")
    print("\n=== Test window Sep 17-30, 2026 (one forecast issued 11pm Sep 16) ===")
    print(table)
    print("\n=== Test MAE by lead day ===")
    print(days)


if __name__ == "__main__":
    main()
