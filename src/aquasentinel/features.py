"""Window features for sensor-fault vs water-quality-event discrimination.

A candidate window W (n x 4, model space) is described relative to a 20-record
baseline B. Only the first DECISION_STEPS records are used: the classifier
must decide within 6 h of the anomaly starting, whatever its eventual length.
"""
from __future__ import annotations

import numpy as np

from .injection import robust_sd

DECISION_STEPS = 24
DEVIANT_Z = 3.0
T, SPC, PH, TURB = range(4)

F4_NAMES = ("n_channels_deviant", "directional_consistency",
            "physical_plausibility", "recovery_smoothness")
F6_NAMES = ("n_channels_deviant", "corroboration", "onset_synchrony",
            "physical_plausibility_ext", "max_step_fraction", "zero_fraction")


def _z(W: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (W - np.median(B, axis=0)) / robust_sd(B)


def _peak_signed(Z: np.ndarray) -> np.ndarray:
    i = np.argmax(np.abs(Z), axis=0)
    return Z[i, np.arange(Z.shape[1])]


def _step_fraction(W: np.ndarray, B: np.ndarray, Z: np.ndarray) -> float:
    c = int(np.argmax(np.abs(Z).max(axis=0)))
    s = np.r_[B[-1, c], W[:, c]]
    amp = s.max() - s.min()
    return 0.0 if amp == 0 else float(np.abs(np.diff(s)).max() / amp)


def features_f4(W: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The four features specified a priori in the original design."""
    W = W[:DECISION_STEPS]
    Z = _z(W, B)
    peak = _peak_signed(Z)
    n_dev = int(np.sum(np.abs(Z).max(axis=0) > DEVIANT_Z))
    directional = float(np.sign(peak[TURB]) == np.sign(peak[SPC]) and np.sign(peak[PH]) == -np.sign(peak[TURB])
                        and peak[TURB] != 0)
    plausible = float(W[:, SPC].min() >= 0.15 * np.median(B[:, SPC]))
    return np.array([n_dev, directional, plausible, _step_fraction(W, B, Z)])


def features_f6(W: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Revised, sign-agnostic features (no assumed direction of change)."""
    W = W[:DECISION_STEPS]
    Z = _z(W, B)
    peak_abs = np.minimum(np.abs(Z).max(axis=0), 10.0)
    dev = peak_abs > DEVIANT_Z
    n_dev = int(dev.sum())
    corroboration = float(np.sort(peak_abs)[-2])
    if n_dev >= 2:
        first = np.array([np.flatnonzero(np.abs(Z[:, c]) > DEVIANT_Z)[0] for c in np.flatnonzero(dev)])
        synchrony = float(np.mean(first - first.min() <= 1))
    else:
        synchrony = 0.0
    plausible = float(W[:, SPC].min() >= 0.15 * np.median(B[:, SPC])
                      and W[:, PH].min() >= 3.5 and W[:, PH].max() <= 10.0
                      and np.abs(W[:, T] - np.median(B[:, T])).max() < 8.0)
    zero_fraction = float(np.mean(W == 0.0))
    return np.array([n_dev, corroboration, synchrony, plausible, _step_fraction(W, B, Z), zero_fraction])


FEATURE_SETS = {"F4": (features_f4, F4_NAMES), "F6": (features_f6, F6_NAMES)}
