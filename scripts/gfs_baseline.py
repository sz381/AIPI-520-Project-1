"""Raw GFS as a baseline, next to raw ECMWF, on the hours where both exist.

GFS 12z runs are archived only from 2026-04-02, so this covers fold2_2026 and the test window, not fold1_2025.
It reads the sample table written by notebook 04, so no notebook has to be re-run.

Data: python scripts/fetch_nwp_runs.py --model gfs_seamless --start 2026-04-02 --end 2026-09-16 --hour 12
      copied to data/raw/gfs_seamless_runs_2026-04-02_to_2026-09-16.csv (the 2026-09-14 run is not in the archive)
Run:  python scripts/gfs_baseline.py
"""

import numpy as np
import pandas as pd

import mos_lib as lib

GFS_FILE = lib.PROJECT_ROOT / "data" / "raw" / "gfs_seamless_runs_2026-04-02_to_2026-09-16.csv"
SAMPLES_FILE = lib.PROJECT_ROOT / "data" / "processed" / "model_samples_2024-03-14_to_2026-09-16.csv.gz"
OBS_FILE = lib.PROJECT_ROOT / "data" / "external" / "rdu_obs_2026-09-17_to_2026-09-30.csv"
FOLD = "fold2_2026"
FINAL_RUN_DAY = pd.Timestamp("2026-09-16")
MODELS = {"N0 raw ECMWF": "ecmwf_12z", "G0 raw GFS": "gfs_12z"}


def samples_with_gfs():
    """The sample table plus the GFS 12z forecast of the same run day, paired like ECMWF (obs hh:51 <-> hh+1:00)."""
    samples = pd.read_csv(SAMPLES_FILE, parse_dates=["run_day", "target_utc"])
    gfs = pd.read_csv(GFS_FILE, usecols=["run_utc", "valid_utc", "temperature_2m"])
    gfs["run_day"] = pd.to_datetime(gfs["run_utc"], utc=True).dt.tz_localize(None).dt.normalize()
    gfs["target_utc"] = pd.to_datetime(gfs["valid_utc"], utc=True) - pd.Timedelta(hours=lib.NWP_VALID_OFFSET_H)
    gfs = gfs.rename(columns={"temperature_2m": "gfs_12z"})[["run_day", "target_utc", "gfs_12z"]]
    return samples.merge(gfs, on=["run_day", "target_utc"], how="left")


def score(rows, actual):
    """Scores of each raw forecast on the rows where the observation and both forecasts exist."""
    ok = actual.notna() & rows[list(MODELS.values())].notna().all(axis=1)
    overall, by_day = {}, {}
    for name, column in MODELS.items():
        error = (actual[ok] - rows.loc[ok, column]).to_numpy()
        overall[name] = {"MAE": lib.mae(error), "RMSE": lib.rmse(error), "bias": float(np.mean(error)), "hours": int(ok.sum())}
        by_day[name] = pd.Series(np.abs(error)).groupby(rows.loc[ok, "lead_day"].to_numpy()).mean()
    return pd.DataFrame(overall).T, pd.DataFrame(by_day).T


def main():
    samples = samples_with_gfs()
    _, val = lib.fold_split(samples, FOLD)
    val_scores, val_by_day = score(val, val["temperature"])

    test = samples[samples["run_day"].eq(FINAL_RUN_DAY)].reset_index(drop=True)
    assert len(test) == lib.HORIZON and test["gfs_12z"].notna().all(), "the Sep 16 12z GFS run must cover all 336 hours"
    obs = pd.read_csv(OBS_FILE, index_col="hour_utc", parse_dates=["hour_utc"])
    observed = obs["temperature"] if "temperature" in obs else obs["observed_deg_c"]
    test_scores, test_by_day = score(test, pd.Series(observed.reindex(test["target_utc"]).to_numpy()))

    scores = pd.concat({FOLD: val_scores, "test Sep 17-30": test_scores}, names=["period", "model"]).round(3)
    by_day = pd.concat({FOLD: val_by_day, "test Sep 17-30": test_by_day}, names=["period", "model"]).round(2)
    by_day.columns.name = "lead_day"
    scores.to_csv(lib.RESULTS_DIR / "gfs_baseline_scores.csv")
    by_day.to_csv(lib.RESULTS_DIR / "gfs_baseline_mae_by_lead_day.csv")

    pd.set_option("display.width", 220)
    print("Raw GFS against raw ECMWF, degC, error = observed - forecast, same hours for both\n")
    print(scores)
    print("\nMAE by lead day\n")
    print(by_day)


if __name__ == "__main__":
    main()
