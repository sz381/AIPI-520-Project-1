"""Build the processed, model-ready tables from the raw downloads.

Run from anywhere (after download_data.py and download_nwp.py):
    python scripts/build_dataset.py

Output in data/processed/:
    obs_hourly.parquet       hourly observations as reported, with quality classes, 2015-01-01 .. cutoff
    climatology.parquet      2015-2023 mean temperature by day of year and UTC hour
    forecast_pairs.parquet   one row per (12z forecast run, target hour)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_modeling import config  # noqa: E402
from weather_modeling.climatology import fit_climatology, save_climatology  # noqa: E402
from weather_modeling.dataset import TARGET, build_pairs  # noqa: E402
from weather_modeling.nwp import MODELS, load_nwp  # noqa: E402
from weather_modeling.obs import build_obs  # noqa: E402


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    obs = build_obs()
    obs.to_parquet(config.OBS_PATH, index=False)
    print(f"observations: {len(obs)} hours, {obs['obs_time_utc'].min()} .. {obs['obs_time_utc'].max()} UTC")
    counts = obs["temp_quality"].value_counts()
    print("temperature quality: " + ", ".join(f"{count} {name}" for name, count in counts.items()))

    clim = fit_climatology(obs)
    save_climatology(clim)

    forecasts = {model: load_nwp(model) for model in MODELS}
    for model, frame in forecasts.items():
        runs = frame["run_time_utc"]
        print(f"{model}: {runs.nunique()} runs, {runs.min():%Y-%m-%d} .. {runs.max():%Y-%m-%d}")

    pairs = build_pairs(obs, clim, forecasts)
    pairs.to_parquet(config.PAIRS_PATH, index=False)
    with_target = pairs[TARGET].notna()
    print(f"pairs: {len(pairs)} rows, {pairs['run_time_utc'].nunique()} runs, {with_target.sum()} rows with a target")
    for model in MODELS:
        usable = with_target & pairs[f"{model}_t2m"].notna()
        print(f"  trainable with {model}: {usable.sum()} rows, {pairs.loc[usable, 'run_time_utc'].nunique()} runs")


if __name__ == "__main__":
    main()
