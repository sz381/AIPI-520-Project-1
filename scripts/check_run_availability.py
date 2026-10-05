"""When each forecast run we use was actually published.

Three times must be kept apart:

- initialisation: when the model run starts (``run_time_utc``, always 12:00 UTC here);
- publication: when its output is released, hours later;
- valid time: the hour a forecast is for.

The Single Runs archive is keyed by initialisation time. That alone does not
show a run was out before a cutoff. The pipeline assumes every run is available
16 hours after initialisation (``config.OBS_KNOWN_LEAD``), which for the real
forecast is the cutoff, Sep 17 00:00 Eastern. This script records the evidence
for that assumption: the upload time of each run's last needed forecast step on
the public storage of NOAA (GFS) and ECMWF (open data).

An upload time is the last time the file was written, so it can only overstate
how late a run was published, never understate it.

Run from anywhere:
    python scripts/check_run_availability.py

Output: reports/run_availability.csv
"""

import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_modeling import config  # noqa: E402
from weather_modeling.nwp import EXCLUDED_RUNS, load_nwp  # noqa: E402


# The target window ends at lead hour 352. At that range GFS is stored every 3
# hours and ECMWF every 6, so f354 and 360h are the last files a forecast needs.
def gfs_file(run):
    return ("https://noaa-gfs-bdp-pds.s3.amazonaws.com",
            f"gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f354")


def ecmwf_file(run, step=360):
    return ("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
            f"{run:%Y%m%d}/{run:%H}z/ifs/0p25/oper/{run:%Y%m%d%H}0000-{step}h-oper-fc.grib2")


def upload_time(bucket, key):
    """Last-modified time of one public file, or NaT if it is not there."""
    try:
        with urllib.request.urlopen(f"{bucket}/?list-type=2&prefix={key}", timeout=60) as response:
            listing = response.read().decode("utf-8", "replace")
    except OSError:
        return pd.NaT
    match = re.search(rf"<Key>{re.escape(key)}</Key><LastModified>(.*?)</LastModified>", listing)
    return pd.Timestamp(match.group(1)).tz_localize(None) if match else pd.NaT


def gfs_published(run):
    bucket, key = gfs_file(run)
    return key, upload_time(bucket, key)


def ecmwf_published(run):
    """The 360h file, or the 240h one: ECMWF's open data stopped at 240h until 2024-11-11."""
    for step in (360, 240):
        bucket, key = ecmwf_file(run, step)
        time = upload_time(bucket, key)
        if pd.notna(time):
            break
    return key, time


SOURCES = {"gfs": gfs_published, "ecmwf": ecmwf_published}


def main():
    rows = []
    for model, published in SOURCES.items():
        runs = pd.DatetimeIndex(sorted(load_nwp(model, include_excluded=True)["run_time_utc"].unique()))
        excluded = pd.to_datetime(EXCLUDED_RUNS.get(model, []))
        with ThreadPoolExecutor(8) as executor:
            found = list(executor.map(published, runs))
        for run, (key, time) in zip(runs, found):
            rows.append({"model": model, "run_time_utc": run, "used": run not in excluded,
                         "file": key, "published_utc": time})
    table = pd.DataFrame(rows)
    table["hours_after_init"] = (table["published_utc"] - table["run_time_utc"]) / pd.Timedelta(hours=1)
    assumed = table["run_time_utc"] + pd.Timedelta(hours=config.OBS_KNOWN_LEAD)
    table["published_before_assumed"] = table["published_utc"] <= assumed

    out = config.PROJECT_ROOT / "reports" / "run_availability.csv"
    table.round(2).to_csv(out, index=False, date_format="%Y-%m-%d %H:%M:%S")

    pd.set_option("display.width", 200)
    print(f"A run is assumed available {config.OBS_KNOWN_LEAD}h after initialisation.\n")
    for model, part in table.groupby("model", sort=False):
        used = part[part["used"]]
        lag = used["hours_after_init"]
        print(f"{model}: {len(used)} runs used, upload time found for {used['published_utc'].notna().sum()}")
        print(f"  published {lag.median():.1f}h after initialisation (median), {lag.max():.1f}h at the latest")
        print(f"  used runs uploaded later than assumed: {(~used['published_before_assumed']).sum()}")
        for _, row in part[~part["used"]].iterrows():
            print(f"  excluded: {row['run_time_utc']:%Y-%m-%d}, uploaded {row['hours_after_init']:.1f}h after initialisation")
    issue = table[table["run_time_utc"] == config.ISSUE_RUN].set_index("model")
    print(f"\nThe runs behind the real forecast (initialised {config.ISSUE_RUN}, cutoff {config.CUTOFF_UTC} UTC):")
    for model, row in issue.iterrows():
        margin = (config.CUTOFF_UTC - row["published_utc"]) / pd.Timedelta(hours=1)
        print(f"  {model}: last needed file uploaded {row['published_utc']} UTC, {margin:.1f}h before the cutoff")
    print(f"\nsaved {out.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
