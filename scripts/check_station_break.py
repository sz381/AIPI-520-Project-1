"""Evidence for the July 2025 break in RDU's temperature readings.

Compares the temperature as reported with two references that do not depend on
the station: ECMWF forecasts 12-35h ahead, and the ERA5 reanalysis. Both show
the reported temperature dropping by roughly 0.7-1.0 degC around 2025-07-22.
That means the readings before and after were measured differently; it does
not say which period is right. The primary analysis leaves the readings as
reported; scripts/break_sensitivity.py shows what the alternatives change.

Run from anywhere (after build_dataset.py):
    python scripts/check_station_break.py

Output: data/external/era5_rdu_hourly.csv  (downloaded once; diagnostic only, never a model input)
        reports/station_break_monthly.csv
        reports/figures/station_break.png
"""

import json
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_modeling import config  # noqa: E402
from weather_modeling.evaluation import INK_MUTED, INK_SECONDARY, chart_style, legend_above  # noqa: E402
from weather_modeling.nwp import load_nwp  # noqa: E402
from weather_modeling.obs import load_obs  # noqa: E402
from weather_modeling.splits import FOLDS  # noqa: E402
from weather_modeling.station_break import estimate_offset  # noqa: E402


ERA5_PATH = config.DATA_DIR / "external" / "era5_rdu_hourly.csv"
ERA5_URL = (
    "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}&hourly=temperature_2m&models=era5"
)
# ERA5 is published about five days behind real time, so stop well before the cutoff.
ERA5_PERIODS = [("2015-01-01", "2018-12-31"), ("2019-01-01", "2022-12-31"), ("2023-01-01", "2026-08-31")]
BREAK = config.STATION_BREAK_UTC


def load_era5():
    if not ERA5_PATH.exists():
        parts = []
        for start, end in ERA5_PERIODS:
            url = ERA5_URL.format(lat=config.LATITUDE, lon=config.LONGITUDE, start=start, end=end)
            with urllib.request.urlopen(url, timeout=300) as response:
                hourly = json.load(response)["hourly"]
            parts.append(pd.DataFrame({"valid_time_utc": hourly["time"], "era5_t2m": hourly["temperature_2m"]}))
        pd.concat(parts).to_csv(ERA5_PATH, index=False)
    era5 = pd.read_csv(ERA5_PATH, parse_dates=["valid_time_utc"])
    return era5.set_index("valid_time_utc")["era5_t2m"]


def step_at_break(difference):
    """Mean of (reported - reference) in the 12 months after the break minus the 12 months before.

    Whole years on both sides, so the reference's own seasonal bias cancels.
    """
    year = pd.DateOffset(years=1)
    before = difference[(difference.index >= BREAK - year) & (difference.index < BREAK)]
    after = difference[(difference.index >= BREAK) & (difference.index < BREAK + year)]
    return after.mean() - before.mean()


def main():
    obs = load_obs()
    reported = obs.set_index("valid_time_utc")["obs_temp_c"]
    forecasts = load_nwp("ecmwf")
    ecmwf = forecasts[forecasts["lead_hour"].between(12, 35)].set_index("valid_time_utc")["t2m"]
    differences = {
        "reported - ERA5": (reported - load_era5()).dropna(),
        "reported - ECMWF forecast": (reported.reindex(ecmwf.index) - ecmwf).dropna(),
    }

    monthly = pd.DataFrame({name: diff.resample("MS").mean() for name, diff in differences.items()})
    out = config.PROJECT_ROOT / "reports" / "station_break_monthly.csv"
    monthly.round(3).to_csv(out, index_label="month")
    print("Monthly mean difference (degC), from 2023:\n")
    print(monthly.loc["2023":].round(2).to_string())
    print(f"\nStep at {BREAK:%Y-%m-%d} (12 months after minus 12 months before):")
    for name, diff in differences.items():
        print(f"  {name}: {step_at_break(diff):+.2f} degC")

    # The archive switched data source on 2025-08-27, five weeks after the break.
    # Within one source the step is still there, so the switch is not the cause.
    gap = differences["reported - ECMWF forecast"]
    source = obs.set_index("valid_time_utc")["temp_source"].reindex(gap.index)
    same_source = gap[(source == "343") & (gap.index >= "2025-02-01")]
    print(f"\nWithin source 343 alone, Feb 2025 onward: {same_source[same_source.index < BREAK].mean():+.2f} before "
          f"the break, {same_source[same_source.index >= BREAK].mean():+.2f} after")

    print("\nOffset as it could have been estimated at each point (station_break.estimate_offset):")
    moments = {fold: pd.Timestamp(start) + pd.Timedelta(hours=config.RUN_HOUR_UTC + config.OBS_KNOWN_LEAD)
               for fold, (start, _) in FOLDS.items()}
    moments["final forecast"] = config.CUTOFF_UTC
    for name, as_of in moments.items():
        offset, days = estimate_offset(obs, forecasts, as_of)
        print(f"  {name} (as of {as_of:%Y-%m-%d}): {offset:.2f} degC from {days} days after the break")

    with plt.rc_context(chart_style()):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        shown = monthly.loc["2021":]
        for name, color in zip(shown.columns, ["#2a78d6", "#eb6834"]):
            ax.plot(shown.index, shown[name], label=name, color=color)
        ax.axvline(BREAK, color=INK_MUTED, linewidth=1)
        ax.annotate(
            f"readings drop\n≈ {BREAK:%b} {BREAK.day}, {BREAK.year}",
            xy=(BREAK, shown.max().max()), xytext=(6, 0), textcoords="offset points", color=INK_SECONDARY, va="top",
        )
        ax.set_ylabel("Monthly mean difference (°C)")
        ax.set_title("RDU reported temperature against two independent references")
        legend_above(ax)
        fig.tight_layout()
        figure = config.FIGURES_DIR / "station_break.png"
        fig.savefig(figure)
    print(f"\nsaved {out.relative_to(config.PROJECT_ROOT)} and {figure.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
