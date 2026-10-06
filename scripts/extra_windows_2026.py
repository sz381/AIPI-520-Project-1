"""Extra check of the selected model on six 2026 windows that mimic the task (14 days from 12am, issued 11pm before).

For each window the models are re-trained from scratch on rows whose target time is before the window start,
so nothing from the window (or later) is used.

Run:  python scripts/extra_windows_2026.py
"""

import numpy as np
import pandas as pd

import mos_lib as lib
import run_mos_experiments as exp

pd.set_option("display.width", 200)
WINDOWS = {  # name -> first forecast day (12am local); the forecast is issued at 11pm the day before
    "Jun 1-14": "2026-06-01", "Jun 17-30": "2026-06-17",
    "Jul 1-14": "2026-07-01", "Jul 17-30": "2026-07-17",
    "Aug 1-14": "2026-08-01", "Aug 17-30": "2026-08-17",
}
METHODS = {
    "Persistence (heuristic)": "H1 persistence",
    "Historical average (heuristic)": "H2 historical average",
    "Station-only LR (04)": "R2 station-only LR (04 LR3)",
    "ECMWF raw": "N0 raw ECMWF",
    "ECMWF lagged mean": "N1 raw lagged-ensemble mean",
    "M14 LR-MOS": "M14 M11 with historical-average base",
}


def main():
    samples, _, _ = exp.prepare_samples(verbose=False)
    overall, parts = {}, []
    for window, first_day in WINDOWS.items():
        start_utc = pd.Timestamp(first_day, tz=lib.NY).tz_convert("UTC")
        run_day = pd.Timestamp(first_day) - pd.Timedelta(days=1)
        train = samples[(samples["target_utc"] < start_utc) & samples["temperature"].notna()]
        test = samples[samples["run_day"].eq(run_day)].reset_index(drop=True)
        assert len(test) == lib.HORIZON and train["target_utc"].max() < start_utc
        assert test["target_utc"].min() == start_utc
        fallback = exp.station_lr(train, test)
        overall[window] = {}
        for label, key in METHODS.items():
            pred = exp.MODELS[key](train, test)
            pred = np.where(np.isnan(pred), fallback, pred)
            pred = np.where(np.isnan(pred), test["climatology"].to_numpy(), pred)
            err = test["temperature"].to_numpy() - pred
            ok = np.isfinite(err)
            overall[window][label] = lib.mae(err[ok])
            parts.append(pd.DataFrame({"window": window, "method": label,
                                       "lead_day": test["lead_day"].to_numpy()[ok], "error": err[ok]}))
        print(f"{window}: trained on {len(train):,} rows (targets before {first_day}), "
              f"scored {int(ok.sum())} of 336 hours")

    table = pd.DataFrame(overall).reindex(METHODS)
    table["average of 6"] = table.mean(axis=1)
    print("\n=== MAE (deg C) per window, every model re-trained on data before the window ===")
    print(table.round(2))

    m14 = table.loc["M14 LR-MOS"]
    gain = pd.DataFrame({
        "vs persistence": 100 * (1 - m14 / table.loc["Persistence (heuristic)"]),
        "vs historical average": 100 * (1 - m14 / table.loc["Historical average (heuristic)"]),
    }).T.round(0)
    print("\n=== M14 improvement over the heuristics (% lower MAE) ===")
    print(gain)

    errors = pd.concat(parts)
    by_day = errors.groupby(["method", "lead_day"])["error"].apply(lib.mae).unstack().reindex(METHODS).round(2)
    print("\n=== MAE by forecast day, pooled over the 6 windows ===")
    print(by_day)
    table.round(3).to_csv(lib.RESULTS_DIR / "extra_windows_2026_mae.csv")
    by_day.to_csv(lib.RESULTS_DIR / "extra_windows_2026_mae_by_day.csv")


if __name__ == "__main__":
    main()
