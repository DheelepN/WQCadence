"""Gap-aware exponentially weighted anomaly scorer.

Decay is defined in wall-clock time, not in measurement cycles: a station that
skips g native steps retains r**g of its prior baseline, with r = 0.5**(1/h)
for a half-life of h native steps. A coarse-sampling policy's baseline is
therefore no staler (and no more sensitive) than a fine-sampling policy's.

Each new reading is standardised by an exponentially weighted mean of past
squared prediction errors (x - baseline)^2, i.e. the variance of what the
detector actually compares. Unlike the usual incremental EW variance, whose
steady-state value shrinks as the gap between samples grows, this keeps
E[z^2] close to 1 at every sampling cadence. The error variance uses a longer
half-life (24 native steps, 6 h) so coarse policies still average several
samples into it.
"""
from __future__ import annotations

import numpy as np

# Minimum standard deviation per channel in model space (temp degC, spcond
# uS/cm, pH units, log1p NTU), set from instrument resolution so a perfectly
# flat baseline cannot produce unbounded z-scores.
SD_FLOOR = np.array([0.05, 2.0, 0.02, 0.05])
Z_CLIP = 10.0


class GapAwareEWMA:
    def __init__(self, n_channels: int = 4, half_life_steps: float = 6.0,
                 sd_floor: np.ndarray = SD_FLOOR, var_half_life_steps: float = 24.0):
        self.r = 0.5 ** (1.0 / half_life_steps)
        self.r_var = 0.5 ** (1.0 / var_half_life_steps)
        self.sd_floor = np.asarray(sd_floor, dtype=float)
        self.mean = None
        self.var = np.zeros(n_channels)

    def update(self, x: np.ndarray, gap_steps: int = 1) -> np.ndarray:
        """Score x against the pre-update baseline, then fold x into it.

        Returns the per-channel z-scores (zeros on the first call)."""
        x = np.asarray(x, dtype=float)
        if self.mean is None:
            self.mean = x.copy()
            return np.zeros_like(x)
        sd = np.maximum(np.sqrt(self.var), self.sd_floor)
        z = np.clip((x - self.mean) / sd, -Z_CLIP, Z_CLIP)
        d = x - self.mean
        retain = self.r ** gap_steps
        self.mean = self.mean + (1.0 - retain) * d
        retain_var = self.r_var ** gap_steps
        self.var = retain_var * self.var + (1.0 - retain_var) * d * d
        return z


def score(z: np.ndarray, aggregate: str = "mean") -> float:
    """Multivariate anomaly score.

    "mean" (a priori design): mean absolute per-channel z, which rewards
    coherent multi-channel change. "rms" (exploratory): root-mean-square z,
    which lets a single strongly deviating channel reach the thresholds."""
    if aggregate == "mean":
        return float(np.mean(np.abs(z)))
    if aggregate == "rms":
        return float(np.sqrt(np.mean(z * z)))
    raise ValueError(aggregate)
