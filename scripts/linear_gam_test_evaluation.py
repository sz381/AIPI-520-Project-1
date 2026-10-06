"""Score the frozen linear GAM forecast on the test window (Sep 17-30, 2026) - run once, after linear_gam.py.

Same protocol as notebook 05: the forecast must match the SHA-256 recorded when it was frozen, and the test
observations (data/external/, written by 05) are used for scoring only. All methods are scored on the same
observed hours; the linear regression is the frozen forecast from 04.

Run:  python scripts/linear_gam_test_evaluation.py
Output: scripts/results/linear_gam_test_scores.csv, linear_gam_test_mae_by_day.csv, linear_gam_test_hourly.csv
"""

import hashlib

import numpy as np
import pandas as pd

from linear_gam import FORECAST_FILE, HASH_FILE, PROJECT_ROOT, RESULTS_DIR, SAMPLES_FILE

LR_FORECAST_FILE = PROJECT_ROOT / "reports" / "forecast_2026-09-17_to_2026-09-30_linear_regression.csv"  # from 04
TEST_OBS_FILE = PROJECT_ROOT / "data" / "external" / "rdu_obs_2026-09-17_to_2026-09-30.csv"  # from 05


def main():
    pd.set_option("display.width", 200)
    recorded = HASH_FILE.read_text().split()[0]
    assert hashlib.sha256(FORECAST_FILE.read_bytes()).hexdigest() == recorded, "frozen GAM forecast was modified"
    print(f"Frozen forecast verified: {HASH_FILE.read_text().splitlines()[1]}")

    gam = pd.read_csv(FORECAST_FILE).set_index("hour_local")
    lr = pd.read_csv(LR_FORECAST_FILE).set_index("hour_local")
    samples = pd.read_csv(SAMPLES_FILE, parse_dates=["run_day"])
    final = samples[samples["run_day"].eq(pd.Timestamp("2026-09-16"))].set_index("target_local")
    observed = pd.read_csv(TEST_OBS_FILE).set_index("hour_local")["observed_deg_c"]
    assert list(gam.index) == list(lr.index) == list(final.index), "forecast files must cover the same 336 hours"

    table = pd.DataFrame({
        "lead_day": final["lead_day"],
        "observed": observed.reindex(gam.index),
        "Persistence": final["persistence"],
        "Historical average": gam["historical_average_deg_c"],
        "ECMWF raw": gam["ecmwf_raw_deg_c"],
        "ECMWF lagged mean": final["ecmwf_lagged_mean"],
        "Linear regression": lr["forecast_deg_c"],
        "Linear GAM": gam["forecast_deg_c"],
    }).rename_axis("hour_local")
    methods = list(table.columns[2:])
    scored = table.dropna(subset=["observed"])
    errors = scored[methods].rsub(scored["observed"], axis=0)  # observed - forecast
    print(f"Scored hours: {len(scored)} of {len(table)}")

    summary = pd.DataFrame({"MAE": errors.abs().mean(), "RMSE": np.sqrt((errors ** 2).mean()),
                            "bias (obs - forecast)": errors.mean()})
    summary["GAM improvement (% lower MAE)"] = 100 * (1 - summary.loc["Linear GAM", "MAE"] / summary["MAE"])
    by_day = errors.abs().groupby(scored["lead_day"]).mean().T
    by_day.columns.name = "forecast day"
    print("\n=== Test window Sep 17-30, 2026 (one forecast issued 11pm Sep 16) ===")
    print(summary.round(2).to_string())
    print("\n=== Test MAE by forecast day ===")
    print(by_day.round(2).to_string())

    summary.round(3).to_csv(RESULTS_DIR / "linear_gam_test_scores.csv")
    by_day.round(3).to_csv(RESULTS_DIR / "linear_gam_test_mae_by_day.csv")
    table.to_csv(RESULTS_DIR / "linear_gam_test_hourly.csv")
    print("\nSaved scripts/results/linear_gam_test_{scores,mae_by_day,hourly}.csv")


if __name__ == "__main__":
    main()
