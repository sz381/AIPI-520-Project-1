"""Score final forecasts against the temperatures observed on Sep 17-30.

Run from anywhere:
    python scripts/download_data.py --holdout
    python scripts/score_submission.py reports/predictions/lr.csv reports/predictions/gbm.csv

Each file is a submission written by ``evaluation.make_submission``; its name
(without .csv) labels the model. With no files, only the baselines are scored.

This is the test set. Run it once the forecasts are frozen, for the report, and
do not go back to tune anything against these numbers. Every scoring is logged
with a fingerprint of the forecast file, so a forecast that changed after it was
first scored shows up in the log. Missing observations are not filled in: the
number of hours actually scored and their quality classes are reported.

Output: reports/final_scores.csv
        reports/final_scoring_log.csv   one line per forecast file per run of this script
        reports/figures/final_mae_by_lead_day.png
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_modeling import config  # noqa: E402
from weather_modeling.dataset import TARGET  # noqa: E402
from weather_modeling.evaluation import BASELINES, holdout_frame, plot_by_lead_day  # noqa: E402
from weather_modeling.metrics import score_by  # noqa: E402


LOG_PATH = config.PROJECT_ROOT / "reports" / "final_scoring_log.csv"


def log_scoring(paths, summary, hours_scored):
    """Append one line per forecast file and warn if a file changed since it was first scored."""
    previous = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame(columns=["name", "sha256"])
    entries = []
    for path in paths:
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        earlier = previous.loc[previous["name"] == path.stem, "sha256"]
        if len(earlier) and fingerprint not in set(earlier):
            print(f"WARNING: {path.stem} was scored before with different predictions; "
                  "its forecast changed after seeing test scores.")
        entries.append({
            "scored_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
            "name": path.stem,
            "sha256": fingerprint,
            "hours_scored": hours_scored,
            "mae_degC": round(float(summary.loc[path.stem, "mae"]), 3),
        })
    if entries:
        pd.concat([previous, pd.DataFrame(entries)], ignore_index=True).to_csv(LOG_PATH, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="*", type=Path)
    args = parser.parse_args()

    submissions = {path.stem: pd.read_csv(path, parse_dates=["target_time_local"]) for path in args.submissions}
    frame = holdout_frame(submissions)
    names = [*BASELINES, *submissions]
    observed = frame[TARGET].notna().sum()
    print(f"scoring {observed} of {len(frame)} target hours", end="")
    print("" if observed == len(frame) else " (the rest are not published by NOAA yet; re-run the download later)")
    counts = frame["target_quality"].value_counts()
    print("target quality: " + ", ".join(f"{count} {name}" for name, count in counts.items()))

    tables = {metric: score_by(frame, names, metric=metric) for metric in ("mae", "rmse", "bias")}
    pd.set_option("display.width", 200)
    print("\nMAE (degC) by lead day\n")
    print(tables["mae"].round(2).to_string())
    summary = pd.DataFrame({metric: table.loc["all", names] for metric, table in tables.items()})
    summary.insert(1, "mae_degF", summary["mae"] * 1.8)
    print("\nAll 14 days, lead days weighted equally (degC unless marked)\n")
    print(summary.round(2).to_string())

    log_scoring(args.submissions, summary, int(observed))

    out = config.PROJECT_ROOT / "reports" / "final_scores.csv"
    pd.concat(tables, names=["metric"]).round(3).to_csv(out)
    figure = config.FIGURES_DIR / "final_mae_by_lead_day.png"
    plot_by_lead_day(tables["mae"], "MAE by lead day, Sep 17 to Sep 30, 2026", path=figure)
    print(f"\nsaved {out.relative_to(config.PROJECT_ROOT)} and {figure.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
