"""When each NWP run used here was actually published.

A run's initialisation time (run_utc) is not its publication time, and the Single Runs archive is keyed by the
former. The pipeline assumes a run is out by the 11pm-local origin of its forecast: the 12z run of day d about 16
hours after initialisation, the 00z run of day d about 28 hours after. This script records the evidence: the upload
time of each run's last needed file on the public storage of NOAA (GFS) and ECMWF (open data).

An upload time is the last time the file was written, so it can overstate how late a run was published, never
understate it. ECMWF's public files stopped at 240 h until 2024-11-11; for those runs the 240 h file is used.

Run:  python scripts/check_run_availability.py   ->  scripts/results/run_availability.csv
"""

import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ECMWF_FILE = PROJECT_ROOT / "data" / "raw" / "ecmwf_ifs_runs_2024-03-14_to_2026-09-16.csv"  # from 01
GFS_FILE = PROJECT_ROOT / "data" / "raw" / "gfs_seamless_runs_2026-04-02_to_2026-09-16.csv"  # from 01
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"
DEADLINE_HOURS = {12: 16, 0: 28}   # hours after initialisation by which the run must be out (midnight local)
CUTOFF_UTC = pd.Timestamp("2026-09-17 04:00")
FINAL_RUN = pd.Timestamp("2026-09-16 12:00")


def upload_time(bucket, key, retries=5):
    """Last-modified time (UTC) of one public file, or NaT if the storage does not list it.

    A failed request is retried and finally raised: treating it as "file not there" would silently change the table.
    """
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{bucket}/?list-type=2&prefix={key}", timeout=60) as response:
                listing = response.read().decode("utf-8", "replace")
            break
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    match = re.search(rf"<Key>{re.escape(key)}</Key><LastModified>(.*?)</LastModified>", listing)
    return pd.Timestamp(match.group(1)).tz_localize(None) if match else pd.NaT


def gfs_published(run):
    # the target window ends at lead 352 h; beyond 120 h GFS is stored every 3 h, so f354 is the last file needed
    key = f"gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f354"
    return key, upload_time("https://noaa-gfs-bdp-pds.s3.amazonaws.com", key)


def ecmwf_published(run):
    for step in (360, 240):
        key = f"{run:%Y%m%d}/{run:%H}z/ifs/0p25/oper/{run:%Y%m%d%H}0000-{step}h-oper-fc.grib2"
        published = upload_time("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com", key)
        if pd.notna(published):
            break
    return key, published


def run_times(path):
    runs = pd.read_csv(path, usecols=["run_utc"])["run_utc"].dropna().unique()
    return pd.DatetimeIndex(pd.to_datetime(runs, utc=True, errors="coerce").dropna().tz_localize(None).sort_values())


def main():
    rows = []
    for model, path, published in [("ecmwf_ifs", ECMWF_FILE, ecmwf_published), ("gfs_seamless", GFS_FILE, gfs_published)]:
        runs = run_times(path)
        with ThreadPoolExecutor(8) as executor:
            found = list(executor.map(published, runs))
        rows += [{"model": model, "run_utc": run, "file": key, "published_utc": when} for run, (key, when) in zip(runs, found)]
    table = pd.DataFrame(rows)
    table["hours_after_init"] = (table["published_utc"] - table["run_utc"]) / pd.Timedelta(hours=1)
    table["deadline_hours"] = table["run_utc"].dt.hour.map(DEADLINE_HOURS)
    table["in_time"] = table["hours_after_init"] <= table["deadline_hours"]
    table.round(2).to_csv(RESULTS_DIR / "run_availability.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    for (model, hour), part in table.groupby(["model", table["run_utc"].dt.hour]):
        lag = part.loc[part["in_time"], "hours_after_init"]
        print(f"{model} {hour:02d}z: {len(part)} runs, upload time found for {part['published_utc'].notna().sum()}; "
              f"published {lag.median():.1f} h after initialisation (median), {lag.max():.1f} h at the latest in time")
        late = part[part["published_utc"].notna() & ~part["in_time"]]
        for _, row in late.iterrows():
            print(f"    uploaded later than the {row['deadline_hours']} h assumed: {row['run_utc']:%Y-%m-%d %H}z, "
                  f"{row['hours_after_init']:.1f} h after initialisation")
    print(f"\nThe 12z runs behind the real forecast (cutoff {CUTOFF_UTC} UTC):")
    for _, row in table[table["run_utc"] == FINAL_RUN].iterrows():
        margin = (CUTOFF_UTC - row["published_utc"]) / pd.Timedelta(hours=1)
        print(f"    {row['model']}: last needed file uploaded {row['published_utc']} UTC, {margin:.1f} h before the cutoff")


if __name__ == "__main__":
    main()
