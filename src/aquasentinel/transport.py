"""1D advection-dispersion transport of an instantaneous point release."""
from __future__ import annotations

import numpy as np

AREA = 5.0  # channel cross-section, m^2


def concentration(x, t, x0, t0, v, D, M, area: float = AREA):
    """C(x, t) in g/m^3 (= mg/L) for M grams released at (x0, t0).

    C = M / (A sqrt(4 pi D tau)) * exp(-(x - x0 - v tau)^2 / (4 D tau)),
    tau = t - t0, and C = 0 for tau <= 0."""
    x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    tau = t - t0
    out = np.zeros(x.shape)
    pos = tau > 0
    tp = tau[pos]
    out[pos] = M / (area * np.sqrt(4 * np.pi * D * tp)) * np.exp(-((x[pos] - x0 - v * tp) ** 2) / (4 * D * tp))
    return out
