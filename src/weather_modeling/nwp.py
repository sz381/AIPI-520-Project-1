"""Archived NWP forecast runs for RDU from the Open-Meteo Single Runs API.

A "single run" is the forecast exactly as issued at one initialisation time, so
pairing a run with later observations never leaks future information (unlike
reanalysis or the stitched "historical forecast" products).

Weather data by Open-Meteo.com (CC BY 4.0).
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import config


API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

# Project name -> (Open-Meteo model id, first 12z run in the archive).
# ECMWF 12z runs reach 240h until 2024-07-15 and 360h from 2024-07-16.
MODELS = {
    "ecmwf": ("ecmwf_ifs", "2024-03-14"),
    "gfs": ("gfs_seamless", "2026-04-02"),
}

# Runs left out of every table because they cannot be shown to have been
# published by the time the pipeline assumes (16h after initialisation): their
# public upload is dated 37h and 24h after initialisation
# (scripts/check_run_availability.py).
EXCLUDED_RUNS = {"ecmwf": ["2024-09-17 12:00", "2025-02-24 12:00"]}

# Open-Meteo variable -> short column name. Units: degC, %, m/s, degrees, mm,
# W/m2, hPa. Kept to <= 10 variables so one run costs about one API call.
VARIABLES = {
    "temperature_2m": "t2m",
    "dew_point_2m": "td2m",
    "relative_humidity_2m": "rh2m",
    "cloud_cover": "cloud",
    "wind_speed_10m": "wind",
    "wind_direction_10m": "wdir",
    "precipitation": "precip",
    "shortwave_radiation": "swrad",
    "pressure_msl": "mslp",
}


def run_url(model, run_time):
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(VARIABLES),
        "models": MODELS[model][0],
        "run": run_time.strftime("%Y-%m-%dT%H:%M"),
        "forecast_days": 16,
        "wind_speed_unit": "ms",
    }
    return API_URL + "?" + urllib.parse.urlencode(params, safe=",:")


def fetch_run(model, run_time, cache_dir=config.NWP_CACHE_DIR, retries=6):
    """Return one run's JSON, cached on disk so downloads can be resumed.

    A run missing from the archive comes back as ``{"error": true, ...}`` and
    is cached too: holes in the archive are permanent, not transient.
    """
    path = cache_dir / model / f"{model}_{run_time:%Y%m%dT%H}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    url = run_url(model, run_time)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            if error.code == 400 and "not available" in body:
                payload = json.loads(body)
                break
            if attempt == retries - 1:
                raise
            # 429 means we hit the per-minute limit; wait it out.
            time.sleep(65 if error.code == 429 else 5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def run_to_frame(payload, run_time):
    """One row per forecast hour: run_time_utc, valid_time_utc, lead_hour, variables."""
    hourly = payload["hourly"]
    frame = pd.DataFrame({short: hourly[name] for name, short in VARIABLES.items()}, dtype="float64")
    valid = pd.to_datetime(hourly["time"])
    frame.insert(0, "lead_hour", ((valid - run_time) / pd.Timedelta("1h")).astype(int))
    frame.insert(0, "valid_time_utc", valid)
    frame.insert(0, "run_time_utc", run_time)
    frame = frame[frame["lead_hour"].between(0, config.MAX_LEAD_HOUR)]
    # Hours beyond the run's horizon come back as nulls.
    return frame.dropna(subset=["t2m"])


def download_model(model, start=None, end=config.ISSUE_RUN, workers=4, raw_dir=config.RAW_NWP_DIR):
    """Download every 12z run in [start, end] and write one CSV per run year.

    Returns the list of run times the archive does not have. Runs the archive
    holds only partly (a shorter horizon than usual) are kept as they are.
    """
    if end > config.ISSUE_RUN:
        raise ValueError(f"runs after {config.ISSUE_RUN} were published after the cutoff")
    start = pd.Timestamp(start or MODELS[model][1]).normalize() + pd.Timedelta(hours=config.RUN_HOUR_UTC)
    runs = pd.date_range(start, end, freq="D")

    with ThreadPoolExecutor(workers) as executor:
        payloads = list(executor.map(lambda run: fetch_run(model, run), runs))

    frames, missing = [], []
    for run, payload in zip(runs, payloads):
        frame = None if payload.get("error") else run_to_frame(payload, run)
        # A few archived runs answer without an error but hold only nulls.
        if frame is None or frame.empty:
            missing.append(run)
        else:
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    out_dir = raw_dir / model
    out_dir.mkdir(parents=True, exist_ok=True)
    for year, part in data.groupby(data["run_time_utc"].dt.year):
        path = out_dir / f"{model}_{config.RUN_HOUR_UTC:02d}z_{year}.csv"
        part.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M")
        print(f"saved {path.relative_to(config.PROJECT_ROOT)} ({part['run_time_utc'].nunique()} runs, {len(part)} rows)")
    return missing


def load_nwp(model, raw_dir=config.RAW_NWP_DIR, include_excluded=False):
    """The usable runs of one model: run_time_utc, valid_time_utc, lead_hour, variables.

    ``include_excluded=True`` also returns the downloaded runs in ``EXCLUDED_RUNS``.
    """
    files = sorted((raw_dir / model).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no NWP files for {model}; run scripts/download_nwp.py")
    data = pd.concat(
        (pd.read_csv(path, parse_dates=["run_time_utc", "valid_time_utc"]) for path in files),
        ignore_index=True,
    )
    if data["run_time_utc"].max() > config.ISSUE_RUN:
        raise ValueError(f"{model} contains runs published after the cutoff")
    if not include_excluded:
        excluded = pd.to_datetime(EXCLUDED_RUNS.get(model, []))
        data = data[~data["run_time_utc"].isin(excluded)].reset_index(drop=True)
    return data
