"""Optional feature helpers shared by the models."""

import numpy as np


def add_fourier_features(data, hour_harmonics=2, doy_harmonics=1):
    """Add sin/cos terms for the diurnal cycle (UTC hour) and the annual cycle.

    Linear models need these to treat hour 23 and hour 0, or Dec 31 and Jan 1,
    as neighbours.
    """
    out = data.copy()
    for k in range(1, hour_harmonics + 1):
        angle = 2 * np.pi * k * out["hour_utc"] / 24
        out[f"hour_sin{k}"], out[f"hour_cos{k}"] = np.sin(angle), np.cos(angle)
    for k in range(1, doy_harmonics + 1):
        angle = 2 * np.pi * k * out["doy"] / 365
        out[f"doy_sin{k}"], out[f"doy_cos{k}"] = np.sin(angle), np.cos(angle)
    return out
