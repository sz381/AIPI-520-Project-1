"""Where does the linear MOS model go wrong? Residuals on the TRAINING data of fold 1, grouped by condition.

Systematic patterns (non-zero mean residual in some groups) that a linear model cannot express are the
justification for trying a tree model on the residual.

Run:  python scripts/residual_analysis.py --model "M8 LR/day: lagged + bias + nwp_anomaly"
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mos_lib as lib
import run_mos_experiments as exp


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", default="fold1_2025")
    args = parser.parse_args()

    samples, _, _ = exp.prepare_samples(verbose=False)
    train, _ = lib.fold_split(samples, args.fold)
    train = train.reset_index(drop=True)
    residual = train["temperature"].to_numpy() - exp.MODELS[args.model](train, train)  # in-sample on training only
    frame = train.assign(residual=residual).dropna(subset=["residual"])

    groups = {
        "local hour": frame["target_hour"],
        "lead day": frame["lead_day"],
        "ECMWF cloud cover (%)": pd.cut(frame["nwp_cloud_cover"], [-1, 10, 40, 70, 90, 101]),
        "ECMWF wind speed (km/h)": pd.cut(frame["nwp_wind_speed_10m"], [-1, 5, 10, 15, 20, 200]),
        "ECMWF precipitation (mm/h)": pd.cut(frame["nwp_precipitation"], [-1, 0, 0.5, 2, 100]),
        "ECMWF temperature (deg C)": pd.cut(frame["nwp_temperature_2m"], [-30, 0, 10, 20, 25, 30, 50]),
    }
    lib.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (name, key) in zip(axes.ravel(), groups.items()):
        table = frame.groupby(key, observed=True)["residual"].agg(mean="mean", mae=lambda r: r.abs().mean(), n="size")
        print(f"\n=== Residual by {name} ===")
        print(table.round(3))
        labels = [str(i) for i in table.index]
        ax.bar(labels, table["mean"], color=np.where(table["mean"] > 0, "#bc4b51", "#2a6f97"))
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set(title=name, ylabel="mean residual (obs - LR), deg C")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.suptitle(f"{args.model}: training residuals ({args.fold})", fontsize=11)
    fig.tight_layout()
    out = lib.RESULTS_DIR / "lr_residual_analysis.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved {out.relative_to(lib.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
