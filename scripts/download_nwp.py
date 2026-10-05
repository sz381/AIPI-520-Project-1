"""Download archived ECMWF and GFS 12z forecast runs for RDU from Open-Meteo.

Run from anywhere:
    python scripts/download_nwp.py

Output: data/raw/openmeteo_single_runs/<model>/<model>_12z_<year>.csv
Per-run JSON responses are cached in data/interim/openmeteo_cache/ (not
committed), so an interrupted download resumes where it stopped.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_modeling import nwp  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(nwp.MODELS), choices=list(nwp.MODELS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    for model in args.models:
        print(f"downloading {model} runs from {nwp.MODELS[model][1]}")
        missing = nwp.download_model(model, workers=args.workers)
        if missing:
            print(f"{len(missing)} run(s) not in the archive: " + ", ".join(f"{run:%Y-%m-%d}" for run in missing))


if __name__ == "__main__":
    main()
