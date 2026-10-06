"""Download archived NWP model runs for RDU from the Open-Meteo Single Runs API.

One request per model run. Rows are appended to a long-format CSV (one row per run x valid hour),
so the script can be stopped and resumed. Runs the API reports as unavailable are logged and skipped.

Example:
    python scripts/fetch_nwp_runs.py --model ecmwf_ifs --start 2024-03-14 --end 2026-09-16 --hour 12
"""

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "external" / "nwp"
API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
LATITUDE, LONGITUDE = 35.8922, -78.7819  # RDU (station USW00013722)
VARIABLES = [
    "temperature_2m", "dew_point_2m", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    "precipitation", "pressure_msl", "shortwave_radiation",
]


def fetch_run(model, run_time, forecast_hours, retries=4):
    query = urllib.parse.urlencode({
        "latitude": LATITUDE, "longitude": LONGITUDE, "models": model,
        "run": run_time.strftime("%Y-%m-%dT%H:%M"), "hourly": ",".join(VARIABLES),
        "timezone": "GMT", "forecast_hours": forecast_hours,
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{API_URL}?{query}", timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            if error.code == 400 and "not available" in body:
                return {"error": True, "reason": body}
            wait = 2 ** attempt * 5
        except (urllib.error.URLError, TimeoutError):
            wait = 2 ** attempt * 5
        time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {model} {run_time:%Y-%m-%d %H}Z after {retries} attempts")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ecmwf_ifs")
    parser.add_argument("--start", default="2024-03-14")
    parser.add_argument("--end", default="2026-09-16")
    parser.add_argument("--hour", type=int, default=12, help="run initialisation hour (UTC)")
    parser.add_argument("--forecast-hours", type=int, default=360)
    parser.add_argument("--pause", type=float, default=0.2, help="seconds between requests")
    parser.add_argument("--part", default="", help="optional suffix to write a separate part file (parallel workers)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_part{args.part}" if args.part else ""
    out_file = OUT_DIR / f"{args.model}_{args.hour:02d}z{suffix}.csv"
    skip_file = OUT_DIR / f"{args.model}_{args.hour:02d}z{suffix}_unavailable.txt"

    done = set()
    if out_file.exists():
        with out_file.open(newline="") as f:
            done = {row["run_utc"] for row in csv.DictReader(f)}
    unavailable = set(skip_file.read_text().split()) if skip_file.exists() else set()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    run_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    write_header = not out_file.exists()
    fetched = 0
    with out_file.open("a", newline="") as f_out, skip_file.open("a") as f_skip:
        writer = csv.writer(f_out)
        if write_header:
            writer.writerow(["run_utc", "valid_utc", "lead_hours", *VARIABLES])
        for day in run_days:
            run_time = datetime(day.year, day.month, day.day, args.hour, tzinfo=timezone.utc)
            run_key = run_time.strftime("%Y-%m-%dT%H:%MZ")
            if run_key in done or run_key in unavailable:
                continue
            data = fetch_run(args.model, run_time, args.forecast_hours)
            if data.get("error"):
                f_skip.write(run_key + "\n")
                f_skip.flush()
                print(f"{run_key}: not available")
            else:
                hourly = data["hourly"]
                for i, valid in enumerate(hourly["time"]):
                    values = [hourly[v][i] for v in VARIABLES]
                    if all(x is None for x in values):
                        continue
                    valid_time = datetime.fromisoformat(valid).replace(tzinfo=timezone.utc)
                    lead = int((valid_time - run_time).total_seconds() // 3600)
                    writer.writerow([run_key, valid_time.strftime("%Y-%m-%dT%H:%MZ"), lead,
                                     *["" if x is None else x for x in values]])
                f_out.flush()
                fetched += 1
                if fetched % 50 == 0:
                    print(f"{run_key}: {fetched} runs fetched", flush=True)
            time.sleep(args.pause)
    print(f"Done: {fetched} new runs -> {out_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
