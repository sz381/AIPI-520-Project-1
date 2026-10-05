"""Score the baselines on the validation folds: the numbers every model has to beat.

Run from anywhere (after build_dataset.py):
    python scripts/evaluate_baselines.py

Output: reports/baselines_validation.csv   MAE, RMSE and bias by lead day (degC)
        reports/baselines_overall.csv      14-day MAE under each reporting convention
        reports/figures/baselines_mae_<fold>.png
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_modeling import config  # noqa: E402
from weather_modeling.dataset import load_training_data  # noqa: E402
from weather_modeling.evaluation import BASELINES, cross_validate, plot_by_lead_day, score_folds  # noqa: E402


def overall_by_convention(scored, names):
    """14-day MAE per fold in degC and degF, weighting lead days equally or pooling all hours.

    The same forecasts give different numbers under each convention, so a score
    is only comparable with another one reported the same way.
    """
    rows = []
    for unit, fahrenheit in (("degC", False), ("degF", True)):
        for weighting, overall in (("lead days equal", "equal"), ("hours pooled", "pooled")):
            table = score_folds(scored, names, fahrenheit=fahrenheit, overall=overall)
            totals = table.xs("all", level="lead_day")[names]
            rows.append(totals.assign(unit=unit, weighting=weighting))
    return pd.concat(rows).set_index(["unit", "weighting"], append=True).sort_index()


def main():
    data = load_training_data(models=("ecmwf",))
    scored = cross_validate({}, data)
    names = list(BASELINES)
    tables = {metric: score_folds(scored, names, metric=metric) for metric in ("mae", "rmse", "bias")}

    pd.set_option("display.width", 200)
    print("Validation MAE (degC) by lead day; 'all' weights the 14 lead days equally\n")
    print(tables["mae"].round(2).to_string())
    overall = overall_by_convention(scored, names)
    print("\n14-day MAE under each reporting convention\n")
    print(overall.round(2).to_string())
    quality = scored.groupby("fold", sort=False)["target_quality"].value_counts().unstack(fill_value=0)
    print("\nValidation targets by quality class\n")
    print(quality.to_string())

    reports = config.PROJECT_ROOT / "reports"
    pd.concat(tables, names=["metric"]).round(3).to_csv(reports / "baselines_validation.csv")
    overall.round(3).to_csv(reports / "baselines_overall.csv")
    print("\nsaved reports/baselines_validation.csv and reports/baselines_overall.csv")

    for fold, part in scored.groupby("fold", sort=False):
        first, last = part["run_time_utc"].min(), part["run_time_utc"].max()
        title = f"Baseline MAE by lead day, validation runs {first:%b} {first.day} to {last:%b} {last.day}, {last.year}"
        path = config.FIGURES_DIR / f"baselines_mae_{fold}.png"
        plot_by_lead_day(tables["mae"].loc[fold], title, path=path)
        print(f"saved {path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
