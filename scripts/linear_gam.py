"""Linear GAM: validate it against the linear regression and the baselines, then make and freeze its forecast.

Same approach as the linear regression in notebook 04, on the same table from 03:
- target: how much warmer or colder than the historical average an hour is (temperature - hist_avg);
  forecast = hist_avg + predicted difference
- one model per forecast day (1-14), recent runs weighted more (half-life 180 days)
- the same 12 inputs (FEATURES)
The only change is the form: each input enters through a smooth curve (pygam.LinearGAM: penalised B-splines,
identity link) instead of a straight line. The hour of day is one cyclic spline instead of four sin/cos terms, and
the three slow-moving ECMWF bias terms stay linear. The smoothing is chosen by GCV for each forecast day.

Run (after 03, with pygam installed):  python scripts/linear_gam.py
Then, once, for the test period:        python scripts/linear_gam_test_evaluation.py
notebooks/06_linear_gam.ipynb only reads what these two scripts write.

Output: scripts/results/linear_gam_validation_scores.csv    MAE and RMSE of every method on the two folds
        scripts/results/linear_gam_validation_mae_by_day.csv
        scripts/results/linear_gam_validation_errors.csv.gz  observed - forecast for every scored hour
        scripts/results/linear_gam_windows_mae.csv           the six 2026 windows
        scripts/results/linear_gam_ecmwf_weight.csv          how much of the ECMWF anomaly each forecast day keeps
        scripts/results/linear_gam_smoothing.csv             lambda, effective dof and GCV of each final GAM
        models/linear_gam.pkl                                the final GAMs and linear regressions (not tracked)
        reports/forecast_2026-09-17_to_2026-09-30_linear_gam.csv  + scripts/results/final_forecast_linear_gam.sha256
"""

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_FILE = PROJECT_ROOT / "data" / "processed" / "model_samples_2024-03-14_to_2026-09-16.csv.gz"  # from 03
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"
MODELS_FILE = PROJECT_ROOT / "models" / "linear_gam.pkl"
FORECAST_FILE = PROJECT_ROOT / "reports" / "forecast_2026-09-17_to_2026-09-30_linear_gam.csv"
HASH_FILE = RESULTS_DIR / "final_forecast_linear_gam.sha256"

NY = "America/New_York"
CUTOFF = pd.Timestamp("2026-09-17", tz=NY)
FINAL_RUN_DAY = pd.Timestamp("2026-09-16")
HALF_LIFE_DAYS = 180

# Inputs of the linear regression (04), and the same information as GAM terms
FEATURES = ["nwp_lagged_anomaly", "nwp_anomaly", "nwp_recent_error", "nwp_bias_14d", "nwp_bias_30d", "nwp_bias_60d",
            "anom_now", "anom_24h", "hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2"]
GAM_SMOOTH = ["nwp_lagged_anomaly", "nwp_anomaly", "nwp_recent_error", "anom_now", "anom_24h"]
GAM_LINEAR = ["nwp_bias_14d", "nwp_bias_30d", "nwp_bias_60d"]
GAM_FEATURES = [*GAM_SMOOTH, *GAM_LINEAR, "target_hour"]  # target_hour: local clock hour 0-23
N_SPLINES, N_SPLINES_HOUR = 10, 12
LAM_GRID = np.logspace(-2, 4, 7)  # one lambda for all terms at a time, chosen by GCV per forecast day

# Same validation as 04: two season-matched folds and six 2026 windows set up like the real task
FOLDS = {"Fold 1 (2025)": ("2025-08-15", "2025-10-15"), "Fold 2 (2026)": ("2026-08-01", "2026-09-16")}
WINDOWS = {"Jun 1-14": "2026-06-01", "Jun 17-30": "2026-06-17", "Jul 1-14": "2026-07-01",
           "Jul 17-30": "2026-07-17", "Aug 1-14": "2026-08-01", "Aug 17-30": "2026-08-17"}
METHODS = ["Persistence", "Historical average", "ECMWF raw", "ECMWF lagged mean", "Linear regression", "Linear GAM"]


def load_samples():
    samples = pd.read_csv(SAMPLES_FILE, parse_dates=["run_day", "target_utc"])
    samples["target_hour"] = pd.to_datetime(samples["target_local"]).dt.hour
    return samples


def recency_weight(rows, newest):
    return (0.5 ** ((newest - rows["run_day"]).dt.days / HALF_LIFE_DAYS)).to_numpy()


def training_rows(train, day, features):
    rows = train[train["lead_day"].eq(day)].dropna(subset=["temperature", "hist_avg", *features])
    return rows, (rows["temperature"] - rows["hist_avg"]).to_numpy()


def gam_terms():
    from pygam import l, s

    terms = s(0, n_splines=N_SPLINES)
    for i in range(1, len(GAM_SMOOTH)):
        terms += s(i, n_splines=N_SPLINES)
    for i in range(len(GAM_SMOOTH), len(GAM_SMOOTH) + len(GAM_LINEAR)):
        terms += l(i)
    return terms + s(len(GAM_FEATURES) - 1, n_splines=N_SPLINES_HOUR, basis="cp", edge_knots=np.array([0.0, 24.0]))


def fit_gam(train):
    """One LinearGAM per forecast day: (temperature - hist_avg) ~ smooth terms, recent runs weighted more."""
    from pygam import LinearGAM

    newest, models = train["run_day"].max(), {}
    for day in range(1, 15):
        rows, y = training_rows(train, day, GAM_FEATURES)
        gam = LinearGAM(gam_terms())
        gam.gridsearch(rows[GAM_FEATURES].to_numpy(float), y, weights=recency_weight(rows, newest),
                       lam=LAM_GRID, progress=False)
        models[day] = gam
    return models


def fit_lr(train):
    """The linear regression of 04, unchanged: one LinearRegression per forecast day."""
    newest, models = train["run_day"].max(), {}
    for day in range(1, 15):
        rows, y = training_rows(train, day, FEATURES)
        models[day] = LinearRegression().fit(rows[FEATURES].to_numpy(float), y,
                                             sample_weight=recency_weight(rows, newest))
    return models


def predict(models, rows, features):
    prediction = np.full(len(rows), np.nan)
    for day, model in models.items():
        ok = (rows["lead_day"].eq(day) & rows[features].notna().all(axis=1)).to_numpy()
        if ok.any():
            prediction[ok] = rows["hist_avg"].to_numpy()[ok] + model.predict(rows.loc[ok, features].to_numpy(float))
    return prediction


def train_before(samples, start_day):
    start_utc = pd.Timestamp(start_day, tz=NY).tz_convert("UTC")
    train = samples[(samples["target_utc"] < start_utc) & samples["temperature"].notna()]
    assert train["target_utc"].max() < start_utc
    return train, start_utc


def evaluate(samples, first_run_day, last_run_day, start_day):
    """Train on targets before `start_day`, forecast the run days in [first, last]; observed - forecast per hour."""
    train, start_utc = train_before(samples, start_day)
    test = samples[samples["run_day"].between(first_run_day, last_run_day) & samples["temperature"].notna()]
    assert start_utc <= test["target_utc"].min()  # no training target inside the evaluation period
    forecasts = pd.DataFrame({
        "Persistence": test["persistence"],
        "Historical average": test["hist_avg"],
        "ECMWF raw": test["ecmwf_12z"],
        "ECMWF lagged mean": test["ecmwf_lagged_mean"],
        "Linear regression": predict(fit_lr(train), test, FEATURES),
        "Linear GAM": predict(fit_gam(train), test, GAM_FEATURES),
    }, index=test.index)
    keep = forecasts.notna().all(axis=1)  # score only hours where every method has a forecast
    errors = forecasts[keep].rsub(test.loc[keep, "temperature"], axis=0)
    return errors.assign(run_day=test.loc[keep, "run_day"], target_local=test.loc[keep, "target_local"],
                         lead_day=test.loc[keep, "lead_day"], target_hour=test.loc[keep, "target_hour"])


mae = lambda e: e.abs().mean()
rmse = lambda e: np.sqrt((e ** 2).mean())


def ecmwf_weight(gam_models, lr_models, rows):
    """Share of the ECMWF anomaly each forecast day keeps (1 = follow ECMWF, 0 = stay on the historical average).

    LR: its two ECMWF coefficients added up, as in 04. GAM: mean change of the prediction when both ECMWF
    anomaly inputs rise by 1 deg C, over `rows`.
    """
    out = {}
    for day, gam in gam_models.items():
        part = rows[rows["lead_day"].eq(day)].dropna(subset=GAM_FEATURES)
        base = part[GAM_FEATURES].to_numpy(float)
        bumped = base.copy()
        bumped[:, [GAM_FEATURES.index("nwp_anomaly"), GAM_FEATURES.index("nwp_lagged_anomaly")]] += 1.0
        coef = dict(zip(FEATURES, lr_models[day].coef_))
        out[day] = {"Linear regression": coef["nwp_lagged_anomaly"] + coef["nwp_anomaly"],
                    "Linear GAM": float(np.mean(gam.predict(bumped) - gam.predict(base)))}
    return pd.DataFrame(out).T.rename_axis("forecast_day")


def smoothing(gam_models):
    return pd.DataFrame({day: {"lambda": float(np.ravel(m.lam)[0]), "edof": float(m.statistics_["edof"]),
                               "gcv": float(m.statistics_["GCV"])} for day, m in gam_models.items()}).T.rename_axis("forecast_day")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refreeze", action="store_true",
                        help="overwrite a frozen forecast that differs from the new one (only before any test scoring)")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    print(f"Loaded {SAMPLES_FILE.relative_to(PROJECT_ROOT)}: {len(samples):,} rows")

    # 1. Validation folds
    fold_errors = {name: evaluate(samples, first, last, first) for name, (first, last) in FOLDS.items()}
    scores = pd.concat({name: pd.DataFrame({"MAE": mae(e[METHODS]), "RMSE": rmse(e[METHODS])})
                        for name, e in fold_errors.items()}, axis=1)
    by_day = pd.concat({name: e.groupby("lead_day")[METHODS].apply(mae) for name, e in fold_errors.items()},
                       names=["fold"])
    print("\nValidation (MAE / RMSE, deg C)\n")
    print(scores.round(2).to_string())
    gam_mae = scores.xs("MAE", axis=1, level=1).loc["Linear GAM"]
    print("\nLinear GAM: % lower MAE than")
    print(pd.DataFrame({m: 100 * (1 - gam_mae / scores.xs("MAE", axis=1, level=1).loc[m])
                        for m in METHODS[:-1]}).round(0).to_string())

    # 2. Six 2026 windows, each issued at 11pm the evening before its first day
    window_errors = {}
    for name, first_day in WINDOWS.items():
        run_day = pd.Timestamp(first_day) - pd.Timedelta(days=1)
        window_errors[name] = evaluate(samples, run_day, run_day, first_day)
    windows = pd.DataFrame({name: mae(e[METHODS]) for name, e in window_errors.items()})
    windows["Average"] = windows.mean(axis=1)
    print("\nSix 2026 windows (MAE, deg C)\n")
    print(windows.round(2).to_string())

    scores.round(3).to_csv(RESULTS_DIR / "linear_gam_validation_scores.csv")
    by_day.round(3).to_csv(RESULTS_DIR / "linear_gam_validation_mae_by_day.csv")
    windows.round(3).to_csv(RESULTS_DIR / "linear_gam_windows_mae.csv")
    errors = pd.concat({**fold_errors, **window_errors}, names=["period"]).reset_index("period")
    errors.round({m: 3 for m in METHODS}).to_csv(RESULTS_DIR / "linear_gam_validation_errors.csv.gz", index=False,
                                                 date_format="%Y-%m-%d")

    # 3. Final models on every row with a target before 12am Sep 17; forecast from the Sep 16 12z run
    train_all, cutoff_utc = train_before(samples, CUTOFF.strftime("%Y-%m-%d"))
    final = samples[samples["run_day"].eq(FINAL_RUN_DAY)].reset_index(drop=True)
    gam_models, lr_models = fit_gam(train_all), fit_lr(train_all)
    final["forecast"] = predict(gam_models, final, GAM_FEATURES)
    checks = {
        "trained only on targets before 12am Sep 17": train_all["target_utc"].max() < cutoff_utc,
        "336 hours: 12am Sep 17 - 11pm Sep 30": len(final) == 336 and final["target_utc"].min() == cutoff_utc,
        "a forecast for every hour": final["forecast"].notna().all(),
        "targets of the final run are unknown": final["temperature"].isna().all(),
    }
    print("\n" + pd.Series(checks, name="passed").to_string())
    assert all(checks.values())

    MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODELS_FILE.write_bytes(pickle.dumps({"Linear GAM": gam_models, "Linear regression": lr_models,
                                          "gam_features": GAM_FEATURES, "features": FEATURES}))
    last_year = train_all[train_all["run_day"] > train_all["run_day"].max() - pd.Timedelta(days=365)]
    weight = ecmwf_weight(gam_models, lr_models, last_year)
    weight.round(3).to_csv(RESULTS_DIR / "linear_gam_ecmwf_weight.csv")
    smoothing(gam_models).round(4).to_csv(RESULTS_DIR / "linear_gam_smoothing.csv")
    print("\nShare of the ECMWF anomaly kept, by forecast day\n")
    print(weight.round(2).T.to_string())

    output = pd.DataFrame({
        "hour_local": final["target_local"],            # report at hh:51 is labelled hh
        "forecast_deg_c": final["forecast"].round(2),
        "historical_average_deg_c": final["hist_avg"].round(2),
        "ecmwf_raw_deg_c": final["ecmwf_12z"].round(2),
    })
    new_bytes = output.to_csv(index=False, lineterminator="\n").encode()
    new_hash = hashlib.sha256(new_bytes).hexdigest()
    frozen_hash = HASH_FILE.read_text().split()[0] if HASH_FILE.exists() else None
    if frozen_hash not in (None, new_hash) and not args.refreeze:
        raise SystemExit(f"{HASH_FILE.relative_to(PROJECT_ROOT)} holds a different frozen forecast; "
                         "pass --refreeze to replace it (only if it has not been scored on the test period)")
    FORECAST_FILE.write_bytes(new_bytes)
    if frozen_hash != new_hash:  # a re-run that reproduces the frozen forecast keeps its original timestamp
        frozen_at = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        HASH_FILE.write_text(f"{new_hash}  {FORECAST_FILE.name}\nfrozen at {frozen_at} "
                             "(before scoring on Sep 17-30 observations)\n")
    print(f"\nSaved {FORECAST_FILE.relative_to(PROJECT_ROOT)} (sha256 {new_hash[:12]}...) and "
          f"{MODELS_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
