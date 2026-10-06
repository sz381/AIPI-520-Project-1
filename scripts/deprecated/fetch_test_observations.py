"""Download the RDU observations for the test window (12am Sep 17 - 11pm Sep 30, 2026, Eastern) from NOAA GHCNh.

Same selection and temperature cleaning as notebooks 01/02: FM15 routine reports at minute :51, labelled with the
UTC hour they belong to (floor convention), source-specific quality codes, physical limits, and spike checks.
Writes data/external/rdu_obs_2026-09-17_to_2026-09-30.csv. Run only AFTER the final forecast has been frozen.
"""

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = PROJECT_ROOT / "data" / "external" / "rdu_obs_2026-09-17_to_2026-09-30.csv"
URL = ("https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/2026/psv/"
       "GHCNh_USW00013722_2026.psv")
NY = "America/New_York"
TEST_START = pd.Timestamp("2026-09-17", tz=NY).tz_convert("UTC")
TEST_END = pd.Timestamp("2026-10-01", tz=NY).tz_convert("UTC")  # exclusive

ISD_SOURCES = {"313", "314", "315", "322", "335", "343", "344", "346"}
LEGACY_SOURCES = {"220", "221", "222", "223", "347", "348"}


def code(series):
    return series.astype("string").str.replace(r"\.0$", "", regex=True).fillna("")


def spike_size(x):
    jump_in, jump_out = x - x.shift(1), x - x.shift(-1)
    return np.minimum(jump_in.abs(), jump_out.abs()).where(np.sign(jump_in) == np.sign(jump_out), 0)


def main():
    with urllib.request.urlopen(URL, timeout=300) as response:
        raw = pd.read_csv(io.TextIOWrapper(response, encoding="utf-8"), sep="|", low_memory=False)
    raw = raw[raw["temperature_Report_Type"].eq("FM15") & raw["Minute"].eq(51)].copy()
    raw["observation_utc"] = pd.to_datetime(raw["DATE"], utc=True)
    raw["hour_utc"] = raw["observation_utc"].dt.floor("h")
    # one day of margin on both sides so the spike checks have neighbours
    window = raw[(raw["hour_utc"] >= TEST_START - pd.Timedelta(days=1)) & (raw["hour_utc"] < TEST_END + pd.Timedelta(days=1))]
    window = window.drop_duplicates("hour_utc").set_index("hour_utc").sort_index()
    grid = pd.date_range(window.index.min(), window.index.max(), freq="h", name="hour_utc")
    obs = window.reindex(grid)

    for v in ["temperature", "dew_point_temperature", "altimeter"]:
        obs[v] = pd.to_numeric(obs[v], errors="coerce")
    obs["temperature_raw"] = obs["temperature"]
    source, quality = code(obs["temperature_Source_Code"]), code(obs["temperature_Quality_Code"])
    reject = ((source.isin(ISD_SOURCES) & quality.isin({"2", "3", "6", "7"}))
              | (source.isin(LEGACY_SOURCES) & quality.isin({"2", "3", "5"}))
              | quality.str.contains(r"[a-z]", regex=True))
    reject |= ~obs["temperature"].between(-30, 45) & obs["temperature"].notna()
    corrupted = ((spike_size(obs["temperature"]) > 4).astype(int) + (spike_size(obs["dew_point_temperature"]) > 5).astype(int)
                 + (spike_size(obs["altimeter"]) > 3).astype(int)) >= 2
    reject |= corrupted | (spike_size(obs["temperature"]) > 8)
    obs["temperature"] = obs["temperature"].mask(reject)
    obs["rejected"] = reject & obs["temperature_raw"].notna()

    test = obs[(obs.index >= TEST_START) & (obs.index < TEST_END)]
    out = pd.DataFrame({
        "hour_local": test.index.tz_convert(NY).strftime("%Y-%m-%d %H:%M"),
        "temperature": test["temperature"], "temperature_raw": test["temperature_raw"],
        "rejected": test["rejected"], "source": code(test["temperature_Source_Code"]),
    }, index=test.index)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index_label="hour_utc")
    print(f"Saved {OUT_FILE.relative_to(PROJECT_ROOT)}: {len(out)} hours (expected 336), "
          f"{int(out['temperature'].notna().sum())} with temperature, {int(out['rejected'].sum())} rejected, "
          f"sources {out['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
