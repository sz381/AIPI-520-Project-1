"""How much the validation scores depend on the July 2025 break in the station record.

Scores the baselines and one reference regression under three treatments:
readings as reported (the primary analysis), earlier readings shifted by an
offset estimated only from data available at each fold's first cutoff, and
earlier targets dropped from training.

Run from anywhere (after build_dataset.py):
    python scripts/break_sensitivity.py

Output: reports/station_break_sensitivity.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_modeling import config  # noqa: E402
from weather_modeling.dataset import TARGET  # noqa: E402
from weather_modeling.evaluation import break_sensitivity  # noqa: E402
from weather_modeling.nwp import load_nwp  # noqa: E402
from weather_modeling.obs import load_obs  # noqa: E402
from weather_modeling.station_break import estimate_offset  # noqa: E402


FEATURES = ["ecmwf_t2m", "clim_temp_c", "anom_24h"]


def ecmwf_lr(train, val):
    """Reference regression for this check only, not the project's model.

    One fit per lead day on the ECMWF forecast, climatology and 24h anomaly: the
    simplest model that learns an offset between forecast and observation, which
    is what the break distorts.
    """
    predicted = pd.Series(np.nan, index=val.index)
    for lead_day, rows in val.groupby("lead_day"):
        fit = train[train["lead_day"] == lead_day].dropna(subset=FEATURES)
        model = LinearRegression().fit(fit[FEATURES], fit[TARGET])
        predicted.loc[rows.index] = model.predict(rows[FEATURES].fillna({"anom_24h": 0}))
    return predicted.to_numpy()


def main():
    table = break_sensitivity({"ecmwf_lr": ecmwf_lr})
    pd.set_option("display.width", 200)
    print("14-day MAE (degC) on each fold under each treatment of the station break\n")
    print(table.round(2).to_string())

    offset, days = estimate_offset(load_obs(), load_nwp("ecmwf"), config.CUTOFF_UTC)
    print(f"\nOffset estimated at the real cutoff ({config.CUTOFF_UTC}): {offset:.2f} degC from {days} days")

    out = config.PROJECT_ROOT / "reports" / "station_break_sensitivity.csv"
    table.round(3).to_csv(out)
    print(f"saved {out.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
