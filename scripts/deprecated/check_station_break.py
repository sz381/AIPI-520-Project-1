"""Evidence for the July 2025 break in RDU's temperature readings.

Compares the cleaned station temperature with two references that do not depend on the station: ECMWF forecasts
12-35 h ahead, and the ERA5 reanalysis. Both show the reported temperature dropping by roughly 0.7-1.0 degC around
2025-07-22. That means the readings before and after were measured differently; it does not say which period is
right. It is the reason models trained mostly on 2024 to mid-2025 learn an ECMWF bias that no longer holds, and why
recency weighting helps.

ERA5 is a diagnostic here and nothing else: it is never a feature or a target.

Run:  python scripts/check_station_break.py
Output: data/external/era5_rdu_hourly.csv (downloaded once), scripts/results/station_break_monthly.csv,
        scripts/results/station_break.png
"""

import json
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd

import mos_lib as lib

OBS_FILE = lib.PROJECT_ROOT / "data" / "interim" / "rdu_temperature_clean_2015-01-01_to_2026-09-16.csv"
ERA5_FILE = lib.PROJECT_ROOT / "data" / "external" / "era5_rdu_hourly.csv"
ERA5_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=35.8922&longitude=-78.7819"
            "&start_date={start}&end_date={end}&hourly=temperature_2m&models=era5")
# ERA5 is published about five days behind real time, so stop well before the Sep 17 cutoff.
ERA5_PERIODS = [("2015-01-01", "2018-12-31"), ("2019-01-01", "2022-12-31"), ("2023-01-01", "2026-08-31")]
BREAK = pd.Timestamp("2025-07-22")
AS_OF = {"fold1_2025": "2025-08-16 04:00", "fold2_2026": "2026-08-02 04:00", "final forecast": "2026-09-17 04:00"}


def load_era5():
    if not ERA5_FILE.exists():
        parts = []
        for start, end in ERA5_PERIODS:
            with urllib.request.urlopen(ERA5_URL.format(start=start, end=end), timeout=300) as response:
                hourly = json.load(response)["hourly"]
            parts.append(pd.DataFrame({"valid_time_utc": hourly["time"], "era5_t2m": hourly["temperature_2m"]}))
        pd.concat(parts).to_csv(ERA5_FILE, index=False)
    return pd.read_csv(ERA5_FILE, parse_dates=["valid_time_utc"]).set_index("valid_time_utc")["era5_t2m"]


def step_between(gap, days):
    """Mean of (station - reference) over `days` after the break minus the same number of days before it."""
    window = pd.Timedelta(days=days)
    after = gap[(gap.index >= BREAK) & (gap.index < BREAK + window)]
    before = gap[(gap.index >= BREAK - window) & (gap.index < BREAK)]
    return after.mean() - before.mean()


def main():
    obs = pd.read_csv(OBS_FILE, parse_dates=["hour_utc"])
    # the report labelled hh (floor convention) is taken at hh:51, so it is compared with values valid at hh+1:00
    station = pd.Series(obs["temperature"].to_numpy(),
                        index=obs["hour_utc"].dt.tz_localize(None) + pd.Timedelta(hours=lib.NWP_VALID_OFFSET_H))
    nwp = lib.load_nwp()
    nwp = nwp[nwp["lead_hours"].between(12, 35)]
    ecmwf = pd.Series(nwp["temperature_2m"].to_numpy(), index=nwp["valid_utc"].dt.tz_localize(None))
    gaps = {"station - ERA5": (station - load_era5()).dropna(),
            "station - ECMWF forecast": (station.reindex(ecmwf.index) - ecmwf).dropna()}

    monthly = pd.DataFrame({name: gap.resample("MS").mean() for name, gap in gaps.items()})
    monthly.round(3).to_csv(lib.RESULTS_DIR / "station_break_monthly.csv", index_label="month")
    pd.set_option("display.width", 200)
    print("Monthly mean difference (degC), from 2024:\n")
    print(monthly.loc["2024":].round(2))
    print(f"\nStep at {BREAK:%Y-%m-%d}, 12 months after minus 12 months before (a full year cancels the seasonal cycle):")
    for name, gap in gaps.items():
        print(f"    {name}: {step_between(gap, 365):+.2f} degC")
    print("\nThe same step as it could have been estimated from ECMWF at each point, using only earlier data:")
    for name, as_of in AS_OF.items():
        as_of = pd.Timestamp(as_of)
        days = min((as_of - BREAK).days, 365)
        gap = gaps["station - ECMWF forecast"]
        print(f"    {name} (as of {as_of:%Y-%m-%d}): {step_between(gap[gap.index < as_of], days):+.2f} degC from {days} days")

    surface, ink, muted, grid = "#fcfcfb", "#52514e", "#898781", "#e1e0d9"
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=surface)
    ax.set_facecolor(surface)
    shown = monthly.loc["2021":]
    for name, color in zip(shown.columns, ["#2a78d6", "#eb6834"]):
        ax.plot(shown.index, shown[name], label=name, color=color, linewidth=2)
    ax.axvline(BREAK, color=muted, linewidth=1)
    ax.annotate(f"readings drop\n≈ {BREAK:%b} {BREAK.day}, {BREAK.year}", xy=(BREAK, shown.max().max()),
                xytext=(6, 0), textcoords="offset points", color=ink, va="top")
    ax.set_ylabel("Monthly mean difference (°C)", color=ink)
    ax.set_title("RDU reported temperature against two independent references", loc="left", fontweight="semibold", pad=30)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=ink, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(lib.RESULTS_DIR / "station_break.png", dpi=200, facecolor=surface)
    print(f"\nSaved scripts/results/station_break_monthly.csv and scripts/results/station_break.png")


if __name__ == "__main__":
    main()
