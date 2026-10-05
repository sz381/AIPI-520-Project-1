"""Project-wide constants: paths, station, and the geometry of one forecast issue.

Times are tz-naive UTC unless a name ends in ``_local`` (US Eastern wall clock).
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_OBS_DIR = DATA_DIR / "raw" / "ghcnh_rdu"
RAW_HOLDOUT_DIR = DATA_DIR / "raw" / "ghcnh_rdu_holdout"
RAW_NWP_DIR = DATA_DIR / "raw" / "openmeteo_single_runs"
NWP_CACHE_DIR = DATA_DIR / "interim" / "openmeteo_cache"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "reports" / "predictions"

OBS_PATH = PROCESSED_DIR / "obs_hourly.parquet"
CLIMATOLOGY_PATH = PROCESSED_DIR / "climatology.parquet"
PAIRS_PATH = PROCESSED_DIR / "forecast_pairs.parquet"

LOCAL_TZ = "America/New_York"
LATITUDE, LONGITUDE = 35.8922, -78.7819  # GHCNh station USW00013722

# The task: hourly temperature for Sep 17 12am .. Sep 30 11pm (Eastern), using
# only data from before Sep 17 12am Eastern (= 04:00 UTC).
TARGET_START_LOCAL = pd.Timestamp("2026-09-17 00:00")
TARGET_END_LOCAL = pd.Timestamp("2026-09-30 23:00")
CUTOFF_UTC = pd.Timestamp("2026-09-17 04:00")

# The last NWP run we may use. A run's initialisation time is not its
# publication time: the Sep 16 12z runs were on the public NOAA and ECMWF
# storage at 17:10 and 19:35 UTC that day, hours before the cutoff, while the
# Sep 17 00z runs are initialised before the cutoff but published after it.
# scripts/check_run_availability.py records the upload time of every run used.
ISSUE_RUN = pd.Timestamp("2026-09-16 12:00")
RUN_HOUR_UTC = 12

# Hour labels: a report belongs to the clock hour it was taken in, so the report
# at 00:51 on Sep 17 is labelled Sep 17 00:00.
#
# Every historical 12z run is treated as if it were the real issue: the cutoff
# falls 16h after initialisation and the 14-day target window follows it.
# A report is keyed by the next full hour (xx:51 -> xx+1:00), so the last report
# before the cutoff (23:51 local) has valid time run + 16h, and the first target
# (the 00:51 report) has valid time run + 17h. Every observation used as an
# input is therefore from before the cutoff, and every target from after it.
OBS_KNOWN_LEAD = 16
FIRST_LEAD = OBS_KNOWN_LEAD + 1
N_LEAD_DAYS = 14
LAST_LEAD = FIRST_LEAD + 24 * N_LEAD_DAYS - 1
MAX_LEAD_HOUR = 360  # ECMWF 12z horizon; GFS is truncated to match

# Around 2025-07-22 the station's readings drop by roughly 0.5-1.0 degC relative
# to ECMWF forecasts, ERA5 and neighbouring stations, after sitting higher since
# early 2024. The primary analysis uses every reading as reported;
# station_break.py holds the two sensitivity treatments.
STATION_BREAK_START_UTC = pd.Timestamp("2024-02-01")
STATION_BREAK_UTC = pd.Timestamp("2025-07-22")

# Climatology uses only years that end before the first archived NWP run
# (2024-03-14), so no training target contributes to its own climatology.
CLIM_FIRST_YEAR, CLIM_LAST_YEAR = 2015, 2023
