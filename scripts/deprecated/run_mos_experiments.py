"""Compare heuristics, station-only regression and NWP-based (MOS) models on purged, season-matched folds.

Run:  python scripts/run_mos_experiments.py
Writes score tables to scripts/results/.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

import mos_lib as lib

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ANOMALY_STATES = ["anom_now", "anom_24h", "anom_7d", "anom_30d"]
WEATHER_STATES = ["dewpoint_depression", "pressure_change_24h", "cloud_cover_octas"]
HOUR_TERMS = ["hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2"]


def impute_states(samples):
    """Fill missing start-state values with information available at the same time (no look-ahead)."""
    s = samples
    s["anom_now"] = s["anom_now"].fillna(s["anom_24h"]).fillna(s["anom_7d"]).fillna(0.0)
    s["anom_24h"] = s["anom_24h"].fillna(s["anom_now"])
    s["anom_7d"] = s["anom_7d"].fillna(s["anom_24h"])
    s["anom_30d"] = s["anom_30d"].fillna(s["anom_7d"])
    s["pressure_change_24h"] = s["pressure_change_24h"].fillna(0.0)
    s["dewpoint_depression"] = s["dewpoint_depression"].fillna(3.0)  # fixed constants, not data-derived statistics
    s["cloud_cover_octas"] = s["cloud_cover_octas"].fillna(4.0)
    s["nwp_recent_error"] = s["nwp_recent_error"].fillna(s["nwp_bias_30d"]).fillna(0.0)
    for window in (14, 30, 60):
        s[f"nwp_bias_{window}d"] = s[f"nwp_bias_{window}d"].fillna(0.0)
    return s


def add_derived(samples):
    s = samples
    for state in ANOMALY_STATES:
        for tau in (6, 24, 72):
            s[f"{state}_x_decay{tau}"] = s[state] * np.exp(-s["step"] / tau)
    for state in WEATHER_STATES:
        for tau in (6, 24):
            s[f"{state}_x_decay{tau}"] = s[state] * np.exp(-s["step"] / tau)
    s["nwp_dewpoint_depression"] = s["nwp_temperature_2m"] - s["nwp_dew_point_2m"]
    s["nwp_wind_u"] = -s["nwp_wind_speed_10m"] * np.sin(np.deg2rad(s["nwp_wind_direction_10m"]))
    s["nwp_wind_v"] = -s["nwp_wind_speed_10m"] * np.cos(np.deg2rad(s["nwp_wind_direction_10m"]))
    return s


STATION_DECAY = [f"{st}_x_decay{t}" for st in ANOMALY_STATES for t in (6, 24, 72)] + \
                [f"{st}_x_decay{t}" for st in WEATHER_STATES for t in (6, 24)]


# ---------------------------------------------------------------- models: fit_predict(train, test) -> predictions
def column(name):
    return lambda train, test: test[name].to_numpy()


def station_lr(train, test):
    """Station-only LR from 04 (anomaly target, decayed start state, no intercept)."""
    tr = train.dropna(subset=STATION_DECAY)
    model = LinearRegression(fit_intercept=False).fit(tr[STATION_DECAY], tr["temperature"] - tr["climatology"])
    pred = np.full(len(test), np.nan)
    ok = test[STATION_DECAY].notna().all(axis=1).to_numpy()
    pred[ok] = test["climatology"].to_numpy()[ok] + model.predict(test.loc[ok, STATION_DECAY])
    return pred


def per_lead_day_lr(features, anomaly_target=True, intercept=True, half_life_days=None, window_days=None,
                    base="climatology"):
    """One LinearRegression per lead day (24 target hours each).

    half_life_days: down-weight old training runs (weight halves every half_life_days before the newest run).
    window_days: only train on runs from the last window_days before the newest training run.
    """
    def fit_predict(train, test):
        pred = np.full(len(test), np.nan)
        newest = train["run_day"].max()
        if window_days is not None:
            train = train[train["run_day"] > newest - pd.Timedelta(days=window_days)]
        for day in range(1, 15):
            day_features = features[day] if isinstance(features, dict) else features
            tr = train[train["lead_day"].eq(day)].dropna(subset=day_features)
            if len(tr) < 200:
                continue
            base_tr = tr[base] if anomaly_target else 0.0
            weight = None
            if half_life_days is not None:
                weight = 0.5 ** ((newest - tr["run_day"]).dt.days / half_life_days)
            model = LinearRegression(fit_intercept=intercept).fit(tr[day_features], tr["temperature"] - base_tr,
                                                                  sample_weight=weight)
            ok = (test["lead_day"].eq(day) & test[day_features].notna().all(axis=1)).to_numpy()
            base_te = test[base].to_numpy()[ok] if anomaly_target else 0.0
            pred[ok] = base_te + model.predict(test.loc[ok, day_features])
        return pred
    return fit_predict


def single_lr_lead_interactions(features):
    """One LinearRegression; every feature is interacted with [1, s, s^2], s = step / 336."""
    def design(frame):
        s = frame["step"].to_numpy() / lib.HORIZON
        cols = {}
        for f in features:
            x = frame[f].to_numpy()
            cols[f] = x
            cols[f"{f}*s"] = x * s
            cols[f"{f}*s2"] = x * s ** 2
        return pd.DataFrame(cols, index=frame.index)

    def fit_predict(train, test):
        tr = train.dropna(subset=features)
        model = LinearRegression().fit(design(tr), tr["temperature"] - tr["climatology"])
        pred = np.full(len(test), np.nan)
        ok = test[features].notna().all(axis=1).to_numpy()
        pred[ok] = test["climatology"].to_numpy()[ok] + model.predict(design(test[ok]))
        return pred
    return fit_predict


TREE_FEATURES = ["step", "target_hour", "doy_sin1", "doy_cos1", "climatology", "nwp_temperature_2m", "nwp_anomaly",
                 "nwp_dewpoint_depression", "nwp_cloud_cover", "nwp_wind_speed_10m", "nwp_wind_u", "nwp_wind_v",
                 "nwp_precipitation", "nwp_shortwave_radiation", "nwp_pressure_msl", "nwp_recent_error",
                 "nwp_lagged_mean", "nwp_bias_14d", "nwp_bias_30d", "nwp_bias_60d",
                 *ANOMALY_STATES, *WEATHER_STATES]


def tree_residual(make_model):
    """Tree model on the residual: target = observation - NWP temperature."""
    def fit_predict(train, test):
        tr = train.dropna(subset=["nwp_temperature_2m"])
        model = make_model().fit(tr[TREE_FEATURES], tr["temperature"] - tr["nwp_temperature_2m"])
        pred = np.full(len(test), np.nan)
        ok = test["nwp_temperature_2m"].notna().to_numpy()
        pred[ok] = test["nwp_temperature_2m"].to_numpy()[ok] + model.predict(test.loc[ok, TREE_FEATURES])
        return pred
    return fit_predict


MOS_CORE = ["nwp_anomaly", "nwp_recent_error", "anom_now", "anom_24h", *HOUR_TERMS]
MOS_LAGGED = ["nwp_lagged_anomaly", "nwp_recent_error", "anom_now", "anom_24h", *HOUR_TERMS]
MOS_PLUS = MOS_CORE + ["nwp_dewpoint_depression", "nwp_cloud_cover", "nwp_wind_speed_10m",
                       "nwp_precipitation", "nwp_shortwave_radiation"]

MOS_BIAS = MOS_LAGGED + ["nwp_anomaly", "nwp_bias_14d", "nwp_bias_30d", "nwp_bias_60d"]
SPLIT_FEATURES = {day: (MOS_BIAS if day <= 7 else MOS_LAGGED) for day in range(1, 15)}
MOS_LAGGED_HA = ["nwp_lagged_anomaly_ha", "nwp_recent_error", "anom_now", "anom_24h", *HOUR_TERMS]
MOS_BIAS_HA = MOS_LAGGED_HA + ["nwp_anomaly_ha", "nwp_bias_14d", "nwp_bias_30d", "nwp_bias_60d"]


def average_of(*names):
    def fit_predict(train, test):
        return np.mean([MODELS[n](train, test) for n in names], axis=0)
    return fit_predict

MODELS = {
    # heuristics / references
    "H1 persistence": column("persistence"),
    "H2 historical average": column("hist_avg"),
    "R1 climatology (calendar LR, 03)": column("climatology"),
    "R2 station-only LR (04 LR3)": station_lr,
    "N0 raw ECMWF": column("nwp_temperature_2m"),
    "N1 raw lagged-ensemble mean": column("nwp_lagged_mean"),
    # MOS linear regression, one model per lead day
    "M1 LR/day: nwp_anomaly only": per_lead_day_lr(["nwp_anomaly"]),
    "M3 LR/day: core (+ hour terms)": per_lead_day_lr(MOS_CORE),
    "M4 LR/day: core + other NWP vars": per_lead_day_lr(MOS_PLUS),
    "M6 LR/day: core with lagged-ensemble NWP": per_lead_day_lr(MOS_LAGGED),
    "M8 LR/day: lagged + bias 30d + nwp_anomaly": per_lead_day_lr(MOS_LAGGED + ["nwp_anomaly", "nwp_bias_30d"]),
    "M9 LR/day: lagged + biases 14/30/60d": per_lead_day_lr(MOS_BIAS),
    "M10 M9, recency-weighted (half-life 90 d)": per_lead_day_lr(MOS_BIAS, half_life_days=90),
    "M11 M9, recency-weighted (half-life 180 d)": per_lead_day_lr(MOS_BIAS, half_life_days=180),
    "M12 M9, last 365 days only": per_lead_day_lr(MOS_BIAS, window_days=365),
    "M5 single LR x lead (core)": single_lr_lead_interactions(MOS_CORE),
    "M13 split: days 1-7 M11 features, days 8-14 M6 features (hl 180 d)": per_lead_day_lr(SPLIT_FEATURES, half_life_days=180),
    "M14 M11 with historical-average base": per_lead_day_lr(MOS_BIAS_HA, half_life_days=180, base="hist_avg"),
    "M15 M6 with historical-average base (hl 180 d)": per_lead_day_lr(MOS_LAGGED_HA, half_life_days=180, base="hist_avg"),
    "E1 average of M6 and M11": average_of("M6 LR/day: core with lagged-ensemble NWP",
                                           "M11 M9, recency-weighted (half-life 180 d)"),
    "E2 average of M13 and M15": average_of("M13 split: days 1-7 M11 features, days 8-14 M6 features (hl 180 d)",
                                            "M15 M6 with historical-average base (hl 180 d)"),
    # trees on the residual
    "T1 random forest residual": tree_residual(lambda: RandomForestRegressor(
        n_estimators=150, min_samples_leaf=30, max_features=0.4, n_jobs=-1, random_state=0)),
    "T2 hist gradient boosting residual": tree_residual(lambda: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0,
        random_state=0)),
}


ROLLING_MODELS = ["H2 historical average", "R1 climatology (calendar LR, 03)", "R2 station-only LR (04 LR3)",
                  "N0 raw ECMWF", "N1 raw lagged-ensemble mean", "M6 LR/day: core with lagged-ensemble NWP",
                  "M9 LR/day: lagged + biases 14/30/60d", "M10 M9, recency-weighted (half-life 90 d)",
                  "M11 M9, recency-weighted (half-life 180 d)", "M12 M9, last 365 days only",
                  "M13 split: days 1-7 M11 features, days 8-14 M6 features (hl 180 d)",
                  "M14 M11 with historical-average base", "M15 M6 with historical-average base (hl 180 d)",
                  "E1 average of M6 and M11", "E2 average of M13 and M15"]


def rolling_monthly(samples, first_month="2025-01", last_month="2026-08"):
    """Robustness check: every month is validated by models trained on rows with targets before that month."""
    parts = []
    for month in pd.period_range(first_month, last_month, freq="M"):
        start = month.start_time
        start_utc = pd.Timestamp(start.year, start.month, 1, tz=lib.NY).tz_convert("UTC")
        train = samples[(samples["target_utc"] < start_utc) & samples["temperature"].notna()]
        test = samples[samples["run_day"].dt.to_period("M").eq(month) & samples["temperature"].notna()]
        fallback = station_lr(train, test)
        for name in ROLLING_MODELS:
            pred = MODELS[name](train, test)
            pred = np.where(np.isnan(pred), fallback, pred)
            pred = np.where(np.isnan(pred), test["climatology"].to_numpy(), pred)
            parts.append(pd.DataFrame({"model": name, "fold": str(month), "lead_day": test["lead_day"].to_numpy(),
                                       "error": test["temperature"].to_numpy() - pred}))
    rolled = pd.concat(parts, ignore_index=True)
    monthly = rolled.groupby(["model", "fold"])["error"].apply(lib.mae).unstack("fold").round(2)
    overall = rolled.groupby("model")["error"].agg(MAE=lib.mae, RMSE=lib.rmse).round(3).sort_values("MAE")
    monthly.to_csv(lib.RESULTS_DIR / "rolling_monthly_mae.csv")
    print("\n=== Rolling monthly backtest (MAE per validation month) ===")
    print(monthly.loc[overall.index])
    print("\n=== Rolling monthly backtest, pooled ===")
    print(overall)
    print("\nMAE by lead day:")
    print(lib.by_lead_day(rolled).round(2).loc[overall.index])


def prepare_samples(verbose=True):
    """Samples for every 12z run day with NWP, lagged members, rolling bias and derived features."""
    t0 = time.time()
    hourly = lib.load_hourly()
    nwp = lib.load_nwp("ecmwf_ifs")
    samples = lib.build_samples(hourly, nwp)
    # Time-lagged ensemble: 00z run of day d and 12z run of day d-1, both published before the origin
    members = ["nwp_temperature_2m"]
    if True:  # 00z runs are in the same ECMWF file
        samples = lib.attach_lagged_run(samples, lib.load_nwp("ecmwf_ifs", run_hour=0), 12, "nwp_t2m_00z")
        members.append("nwp_t2m_00z")
    samples = lib.attach_lagged_run(samples, nwp, 24, "nwp_t2m_prev12z")
    members.append("nwp_t2m_prev12z")
    samples["nwp_lagged_mean"] = samples[members].mean(axis=1)
    samples.loc[samples["nwp_temperature_2m"].isna(), "nwp_lagged_mean"] = np.nan  # require the newest run
    samples["nwp_lagged_anomaly"] = samples["nwp_lagged_mean"] - samples["climatology"]
    samples["nwp_lagged_anomaly_ha"] = samples["nwp_lagged_mean"] - samples["hist_avg"]
    samples["nwp_anomaly_ha"] = samples["nwp_temperature_2m"] - samples["hist_avg"]
    for window in (14, 30, 60):
        samples = lib.attach_rolling_bias(samples, hourly, nwp, window_days=window)
    samples = add_derived(impute_states(samples))
    if verbose: print("Lagged members present:", {m: f"{samples[m].notna().mean():.0%}" for m in members},
          "| 30-day bias present:", f"{samples['nwp_bias_30d'].notna().mean():.0%}")
    runs = nwp.groupby("run_utc")["lead_hours"].max()
    if verbose: print(f"NWP runs: {len(runs)} ({runs.index.min():%Y-%m-%d} .. {runs.index.max():%Y-%m-%d}); "
          f"max lead counts: {runs.value_counts().to_dict()}")
    if verbose: print(f"Samples: {len(samples):,} rows, NWP temperature present for {samples['nwp_temperature_2m'].notna().mean():.1%}; "
          f"built in {time.time() - t0:.0f}s")

    return samples, hourly, nwp


def main():
    lib.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples, hourly, nwp = prepare_samples()

    parts = []
    for fold in lib.FOLDS:
        train, test = lib.fold_split(samples, fold)
        fallback = station_lr(train, test)
        print(f"\n{fold}: train {len(train):,} rows (targets < {lib.FOLDS[fold][0]}), "
              f"validate {test['run_day'].nunique()} runs / {len(test):,} rows")
        for name, fit_predict in MODELS.items():
            t1 = time.time()
            pred = fit_predict(train, test)
            missing = np.isnan(pred)
            pred = np.where(missing, fallback, pred)  # rows without NWP (e.g. beyond the run length) -> station LR
            pred = np.where(np.isnan(pred), test["climatology"].to_numpy(), pred)
            parts.append(pd.DataFrame({"model": name, "fold": fold, "lead_day": test["lead_day"].to_numpy(),
                                       "error": test["temperature"].to_numpy() - pred}))
            print(f"  {name:<45} MAE {lib.mae(parts[-1]['error']):.3f}  RMSE {lib.rmse(parts[-1]['error']):.3f}"
                  f"  fallback {missing.mean():.1%}  ({time.time() - t1:.0f}s)")

    predictions = pd.concat(parts, ignore_index=True)
    table = lib.score_table(predictions).round(3)
    lead_mae = lib.by_lead_day(predictions).round(2)
    table.to_csv(lib.RESULTS_DIR / "mos_scores.csv")
    lead_mae.to_csv(lib.RESULTS_DIR / "mos_mae_by_lead_day.csv")
    print("\n=== Scores (sorted by pooled MAE) ===")
    print(table)
    print("\n=== MAE by lead day (pooled over both folds) ===")
    print(lead_mae.loc[table.index])
    rolling_monthly(samples)


if __name__ == "__main__":
    main()
