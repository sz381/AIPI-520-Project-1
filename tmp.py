"""Download a small sample of RDU (Raleigh-Durham airport) hourly observations
for 2026-09-01 .. 2026-09-16 (US Eastern) from two sources, saved as CSV next to this file.

  1. Iowa Environmental Mesonet (IEM) ASOS  ->  rdu_iem_asos_<dates>.csv
     Website:    https://mesonet.agron.iastate.edu/request/download.phtml?network=NC_ASOS
     API params: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help
  2. NOAA NCEI GHCNh (Global Historical Climatology Network hourly)  ->  rdu_ghcnh_<dates>.csv
     Website:    https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly
     Docs:       https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf

Standard library only (no pandas needed):  python3 tmp.py
"""
import csv
import io
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
START = date(2026, 9, 1)  # inclusive, US Eastern
END = date(2026, 9, 17)   # exclusive: last hourly obs is Sep 16 23:51, before the Sep 17 12am cutoff
OUT_DIR = Path(__file__).resolve().parent
TAG = f"{START}_to_{END - timedelta(days=1)}"


def download_iem():
    params = {
        "station": "RDU",
        "data": "all",             # every field, including the raw METAR text
        "year1": START.year, "month1": START.month, "day1": START.day,
        "year2": END.year, "month2": END.month, "day2": END.day,
        "tz": "America/New_York",  # `valid` column is Eastern local time
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "empty",        # blank instead of "M" so numeric columns stay numeric
        "trace": "0.0001",         # trace precipitation as 0.0001 instead of "T"
        "direct": "no",
        "report_type": "3",        # routine hourly METARs only (no specials)
    }
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urllib.parse.urlencode(params)
    out = OUT_DIR / f"rdu_iem_asos_{TAG}.csv"
    with urllib.request.urlopen(url, timeout=120) as resp:
        out.write_bytes(resp.read())
    print("saved", out.name)


def download_ghcnh():
    # USW00013722 = RALEIGH AP; one pipe-separated file per station per year, times in UTC, temps in °C
    url = ("https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/"
           f"by-year/{START.year}/psv/GHCNh_USW00013722_{START.year}.psv")
    start = datetime(START.year, START.month, START.day, tzinfo=NY)
    end = datetime(END.year, END.month, END.day, tzinfo=NY)
    rows = []
    with urllib.request.urlopen(url, timeout=300) as resp:
        reader = csv.DictReader(io.TextIOWrapper(resp, encoding="utf-8", newline=""), delimiter="|")
        fields = reader.fieldnames
        for r in reader:
            # routine hourly METAR: GHCNh also tags a few off-hour specials as FM15, so require :51 too
            if r["temperature_Report_Type"] != "FM15" or r["Minute"] != "51":
                continue
            t_local = datetime.fromisoformat(r["DATE"]).replace(tzinfo=timezone.utc).astimezone(NY)
            if start <= t_local < end:
                r["DATE_local"] = t_local.strftime("%Y-%m-%d %H:%M")
                rows.append(r)

    # drop columns that are empty in every sampled row; put local time next to the UTC DATE
    keep = [c for c in fields if any(r[c] != "" for r in rows)]
    keep.insert(keep.index("DATE") + 1, "DATE_local")
    out = OUT_DIR / f"rdu_ghcnh_{TAG}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out.name} ({len(rows)} rows, {len(keep)} of {len(fields)} columns)")


if __name__ == "__main__":
    download_iem()
    download_ghcnh()
