"""Validate the linear GAM against the LR (same inputs) and the baselines, then write the final forecast.

Run from anywhere (after build_dataset.py, with pygam installed):
    python scripts/evaluate_linear_gam.py

This script does all the fitting and scoring; notebooks/06_linear_gam.ipynb only
reads its outputs to plot and compare them.

Steps
1. Builds the MOS table: the pairs table plus the LR baseline's 12 inputs.
2. Scores ``linear_gam``, ``lr_mos`` (the LR baseline on the same rows) and the
   baselines on the two validation folds and on six 2026 windows laid out like
   the real task. Every model is trained only on targets observed before the
   first target it is scored on (``splits.split_by_run``, purged).
3. Trains both models on every row, saves them, and writes the Sep 17-30
   forecasts with their SHA-256 fingerprints, before anything is scored against
   the test period.

Output: reports/linear_gam_validation.csv         MAE, RMSE, bias by lead day, folds and windows (degC)
        reports/linear_gam_overall.csv            14-day MAE per fold, lead days equal and hours pooled
        reports/linear_gam_windows.csv            14-day MAE of each 2026 window (hours pooled)
        reports/linear_gam_validation_predictions.csv.gz   out-of-fold predictions, one row per scored hour
        reports/linear_gam_ecmwf_weight.csv       share of the ECMWF anomaly kept, GAM and LR, by lead day
        reports/linear_gam_smoothing.csv          lambda, effective dof and GCV of each final GAM
        models/linear_gam.pkl                     the final GAMs and LRs (not tracked by git)
        reports/predictions/linear_gam.csv, lr_mos.csv, frozen_forecasts.csv

Then, once, for the test period:
    python scripts/score_submission.py reports/predictions/linear_gam.csv reports/predictions/lr_mos.csv
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_modeling import config  # noqa: E402
from weather_modeling.dataset import TARGET  # noqa: E402
from weather_modeling.evaluation import BASELINES, cross_validate, make_submission, score_folds  # noqa: E402
from weather_modeling.gam import (  # noqa: E402
    GAM_FEATURES,
    LR_FEATURES,
    TASK_WINDOWS_2026,
    build_mos_table,
    ecmwf_weight,
    fit_gam,
    fit_lr,
    forecast_rows,
    gam_fit_predict,
    lr_ecmwf_weight,
    lr_fit_predict,
    predict,
    save_models,
    smoothing_summary,
    trainable,
)
from weather_modeling.splits import FOLDS  # noqa: E402


MODELS = {"lr_mos": lr_fit_predict, "linear_gam": gam_fit_predict}
METRICS = ("mae", "rmse", "bias")
REPORTS = config.PROJECT_ROOT / "reports"
PREDICTION_COLUMNS = ["fold", "run_time_utc", "valid_time_utc", "target_time_local", "lead_day", "hour_local",
                      TARGET, "target_quality"]


def overall_by_convention(scored, names):
    rows = []
    for weighting, overall in (("lead days equal", "equal"), ("hours pooled", "pooled")):
        totals = score_folds(scored, names, overall=overall).xs("all", level="lead_day")[names]
        rows.append(totals.assign(weighting=weighting))
    return pd.concat(rows).set_index("weighting", append=True).sort_index()


def improvement(table, model, against):
    """Percent lower MAE of ``model`` than each column in ``against``."""
    return pd.DataFrame({name: 100 * (1 - table[model] / table[name]) for name in against}).round(0)


def saved(path):
    print(f"saved {path.relative_to(config.PROJECT_ROOT)}")


def main():
    pd.set_option("display.width", 200)
    table = build_mos_table()
    data = trainable(table)
    names = [*BASELINES, *MODELS]
    print(f"MOS table: {len(data):,} trainable rows from {data['run_time_utc'].nunique()} runs")

    # 1. Validation folds
    scored = cross_validate(MODELS, data, FOLDS)
    tables = {metric: score_folds(scored, names, metric=metric) for metric in METRICS}
    overall = overall_by_convention(scored, names)
    print("\nValidation MAE (degC) by lead day; 'all' weights the 14 lead days equally\n")
    print(tables["mae"].round(2).to_string())
    print("\n14-day MAE, both weightings\n")
    print(overall.round(2).to_string())
    totals = tables["mae"].xs("all", level="lead_day")
    print("\nlinear_gam: % lower MAE than\n")
    print(improvement(totals, "linear_gam", ["persistence", "climatology", "raw_ecmwf", "lr_mos"]).to_string())

    # 2. The six 2026 windows (one run each)
    windows = cross_validate(MODELS, data, TASK_WINDOWS_2026)
    window_mae = score_folds(windows, names, overall="pooled").xs("all", level="lead_day")[names]
    window_mae.loc["Average"] = window_mae.mean()
    print("\nSix 2026 windows, MAE (degC, hours pooled)\n")
    print(window_mae.round(2).to_string())

    window_tables = {metric: score_folds(windows, names, metric=metric) for metric in METRICS}
    validation = pd.concat({m: pd.concat([tables[m], window_tables[m]]) for m in METRICS}, names=["metric"])
    outputs = {
        "linear_gam_validation.csv": validation,
        "linear_gam_overall.csv": overall,
        "linear_gam_windows.csv": window_mae.rename_axis("window"),
    }
    for name, frame in outputs.items():
        frame.round(3).to_csv(REPORTS / name)
        saved(REPORTS / name)
    predictions = pd.concat([scored, windows], ignore_index=True)[[*PREDICTION_COLUMNS, *names]]
    path = REPORTS / "linear_gam_validation_predictions.csv.gz"
    predictions.round({name: 3 for name in [TARGET, *names]}).to_csv(path, index=False, date_format="%Y-%m-%d %H:%M")
    saved(path)

    # 3. Final models on every row, forecast for Sep 17-30, frozen before any test scoring
    final = forecast_rows(table)
    gam_models, lr_models = fit_gam(data), fit_lr(data)
    saved(save_models(gam_models, lr_models))
    forecasts = {
        "linear_gam": predict(gam_models, final, GAM_FEATURES),
        "lr_mos": predict(lr_models, final, LR_FEATURES),
    }
    frozen = []
    for name, values in forecasts.items():
        path = make_submission(values, name)
        frozen.append({"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "frozen_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")})
        saved(path)
    frozen = pd.DataFrame(frozen)
    frozen.to_csv(config.PREDICTIONS_DIR / "frozen_forecasts.csv", index=False)
    print(frozen.to_string(index=False))

    last_year = data[data["run_time_utc"] >= data["run_time_utc"].max() - pd.Timedelta(days=365)]
    weight = pd.DataFrame({"linear_gam": ecmwf_weight(gam_models, last_year), "lr_mos": lr_ecmwf_weight(lr_models)})
    weight.rename_axis("lead_day").round(3).to_csv(REPORTS / "linear_gam_ecmwf_weight.csv")
    saved(REPORTS / "linear_gam_ecmwf_weight.csv")
    print("\nShare of the ECMWF anomaly kept, by lead day\n")
    print(weight.round(2).T.to_string())

    smoothing = smoothing_summary(gam_models)
    smoothing.round(4).to_csv(REPORTS / "linear_gam_smoothing.csv")
    saved(REPORTS / "linear_gam_smoothing.csv")
    print("\nFinal GAMs: lambda chosen by GCV and effective degrees of freedom\n")
    print(smoothing.round(3).T.to_string())


if __name__ == "__main__":
    main()
