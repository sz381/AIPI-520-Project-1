"""Download RDU GHCNh hourly observations into one directory per local year.

Standard library only. Run from anywhere:
    python scripts/download_data.py

Output: data/raw/ghcnh_rdu/<year>/rdu_ghcnh_<start>_to_<end>.csv
"""

import csv
import io
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


NY = ZoneInfo("America/New_York")
START = date(2015, 1, 1)  # inclusive, US Eastern
END = date(2026, 9, 17)  # exclusive; includes all of September 16
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "ghcnh_rdu"

STATION_ID = "USW00013722"  # Raleigh-Durham International Airport
BASE_URL = (
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
    "hourly/access/by-year/{year}/psv/GHCNh_{station}_{year}.psv"
)


def download_ghcnh():
    start = datetime(START.year, START.month, START.day, tzinfo=NY)
    end = datetime(END.year, END.month, END.day, tzinfo=NY)
    rows_by_year = defaultdict(list)
    field_order = []
    populated_fields = set()

    # Source files use UTC years. Converting to Eastern time can move the first
    # few UTC observations into the preceding local year, so download every
    # source year once and bucket rows by their converted local year.
    for source_year in range(START.year, END.year + 1):
        url = BASE_URL.format(year=source_year, station=STATION_ID)
        print(f"downloading {source_year}: {url}")

        with urllib.request.urlopen(url, timeout=300) as response:
            reader = csv.DictReader(
                io.TextIOWrapper(response, encoding="utf-8", newline=""),
                delimiter="|",
            )
            if not reader.fieldnames:
                raise RuntimeError(f"No header found in {url}")

            for field in reader.fieldnames:
                if field not in field_order:
                    field_order.append(field)

            for row in reader:
                # GHCNh also labels some off-hour special reports FM15, so keep
                # only the routine hourly reports observed at minute 51.
                if (
                    row.get("temperature_Report_Type") != "FM15"
                    or row.get("Minute") != "51"
                ):
                    continue

                utc_time = datetime.fromisoformat(row["DATE"]).replace(
                    tzinfo=timezone.utc
                )
                local_time = utc_time.astimezone(NY)
                if not start <= local_time < end:
                    continue

                row["DATE_local"] = local_time.strftime("%Y-%m-%d %H:%M")
                compact_row = {key: value for key, value in row.items() if value != ""}
                rows_by_year[local_time.year].append(compact_row)
                populated_fields.update(compact_row)

    # Drop columns empty throughout the requested period and put local time
    # immediately after the UTC DATE column.
    keep = [field for field in field_order if field in populated_fields]
    keep.insert(keep.index("DATE") + 1, "DATE_local")

    for year in range(START.year, END.year + 1):
        year_start = max(START, date(year, 1, 1))
        year_end_exclusive = min(END, date(year + 1, 1, 1))
        year_end = year_end_exclusive - timedelta(days=1)
        year_dir = OUT_DIR / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        output_path = year_dir / f"rdu_ghcnh_{year_start}_to_{year_end}.csv"
        rows = rows_by_year[year]

        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=keep,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"saved {output_path.relative_to(PROJECT_ROOT)} "
            f"({len(rows)} rows, {len(keep)} columns)"
        )


if __name__ == "__main__":
    download_ghcnh()
