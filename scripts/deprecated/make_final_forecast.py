"""Train a model on all data before the cutoff and forecast 12am Sep 17 - 11pm Sep 30, 2026.

Inputs available at the cutoff (12am Sep 17 EDT): observations through 23:51 EDT Sep 16, the ECMWF 12z run of
Sep 16 (plus the older 00z Sep 16 and 12z Sep 15 runs for the lagged ensemble).

Run:  python scripts/make_final_forecast.py --model "M8 LR/day: lagged + bias + nwp_anomaly"
"""

import argparse

import numpy as np
import pandas as pd

import mos_lib as lib
import run_mos_experiments as exp

CUTOFF = pd.Timestamp("2026-09-17", tz=lib.NY)
FINAL_RUN_DAY = pd.Timestamp("2026-09-16")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="a key of run_mos_experiments.MODELS")
    args = parser.parse_args()

    samples, hourly, nwp = exp.prepare_samples(verbose=False)
    cutoff_utc = CUTOFF.tz_convert("UTC")
    train = samples[(samples["target_utc"] < cutoff_utc) & samples["temperature"].notna()]
    test = samples[samples["run_day"].eq(FINAL_RUN_DAY)].reset_index(drop=True)

    # Leakage guards
    assert len(test) == lib.HORIZON, f"expected 336 forecast hours, got {len(test)}"
    assert test["target_utc"].min() == cutoff_utc, "first target must be 12am Sep 17 EDT"
    assert (test["origin_utc"] < cutoff_utc).all() and train["target_utc"].max() < cutoff_utc
    assert (test["run_utc"] == pd.Timestamp("2026-09-16 12:00", tz="UTC")).all()
    assert test["temperature"].isna().all(), "targets after the cutoff must be unknown in the feature table"
    assert test["nwp_temperature_2m"].notna().all(), "the Sep 16 12z run must cover all 336 hours"

    prediction = exp.MODELS[args.model](train, test)
    fallback = exp.station_lr(train, test)
    n_fallback = int(np.isnan(prediction).sum())
    prediction = np.where(np.isnan(prediction), fallback, prediction)

    local = test["target_utc"].dt.tz_convert(lib.NY)
    forecast = pd.DataFrame({
        "hour_local": local.dt.strftime("%Y-%m-%d %H:%M"),   # floor convention: hh = observation at hh:51
        "hour_utc": test["target_utc"].dt.strftime("%Y-%m-%dT%H:%MZ"),
        "lead_day": test["lead_day"],
        "temperature_forecast_c": prediction.round(2),
        "climatology_c": test["climatology"].round(2),
        "historical_average_c": test["hist_avg"].round(2),
        "ecmwf_raw_c": test["nwp_temperature_2m"].round(2),
    })
    lib.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = args.model.split()[0]
    out = lib.RESULTS_DIR / f"final_forecast_{slug}.csv"
    forecast.to_csv(out, index=False)
    print(f"Trained on {len(train):,} rows (targets {train['target_utc'].min():%Y-%m-%d} .. "
          f"{train['target_utc'].max().tz_convert(lib.NY):%Y-%m-%d %H:%M %Z}); fallback rows: {n_fallback}")
    print(f"Saved {out.relative_to(lib.PROJECT_ROOT)}")
    print(forecast.groupby("lead_day")[["temperature_forecast_c", "climatology_c", "ecmwf_raw_c"]].agg(["min", "max"]).round(1))


if __name__ == "__main__":
    main()
