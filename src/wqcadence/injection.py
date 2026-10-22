"""Synthetic contamination events and sensor faults, injected in model space.

Amplitudes are expressed in local robust standard deviations (1.4826 * MAD of
the preceding day of quiet data, floored at the detector's SD floor), so the
same severity means the same detectability on a calm or a noisy background.
"""
from __future__ import annotations

import numpy as np

from .detector import SD_FLOOR

SHAPES = ("spike", "ramp_plateau", "damped_oscillation")

# (amplitude low, amplitude high, duration low, duration high) - duration in
# native 15-minute steps: brief 15-45 min, moderate 1-2.5 h, severe 2-5 h.
SEVERITY = {
    "brief": (0.8, 1.3, 1, 3),
    "moderate": (1.5, 2.5, 4, 10),
    "severe": (3.0, 4.5, 8, 20),
}

# Relative direction and weight per channel (temp, spcond, ph, turbidity).
SIGNATURES = {
    # effluent-like discharge: turbidity and conductance up, pH down, slight warming
    "discharge": np.array([0.5, 1.0, -1.0, 1.0]),
    # runoff-like hydrological pulse: turbidity up, conductance diluted, slight cooling
    "runoff": np.array([-0.5, -1.0, -0.5, 1.0]),
}

FAULT_TYPES = ("dropout", "drift", "noise_burst", "collapse", "fouling")


def robust_sd(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    return np.maximum(1.4826 * mad, SD_FLOOR)


def profile(shape: str, n: int) -> np.ndarray:
    """Unit-peak temporal profile of length n (n >= 1)."""
    t = np.arange(n) / max(n - 1, 1)
    if n == 1:
        return np.ones(1)
    if shape == "spike":
        p = np.exp(-4.0 * t)
    elif shape == "ramp_plateau":
        p = np.clip(np.minimum(t / 0.25, (1.0 - t) / 0.25), 0.0, 1.0)
        p = np.maximum(p, 0.25)
    elif shape == "damped_oscillation":
        p = np.exp(-2.5 * t) * (0.6 + 0.4 * np.cos(2 * np.pi * 2.5 * t))
    else:
        raise ValueError(shape)
    return p / p.max()


def draw_event(rng: np.random.Generator, severity: str | None = None,
               shape: str | None = None) -> dict:
    severity = severity or rng.choice(list(SEVERITY))
    shape = shape or rng.choice(SHAPES)
    a_lo, a_hi, d_lo, d_hi = SEVERITY[severity]
    return {"severity": str(severity), "shape": str(shape),
            "amplitude": float(rng.uniform(a_lo, a_hi)),
            "duration": int(rng.integers(d_lo, d_hi + 1))}


def inject_event(X: np.ndarray, onset: int, event: dict, local_sd: np.ndarray,
                 signature: str = "discharge") -> np.ndarray:
    out = X.copy()
    n = event["duration"]
    bump = profile(event["shape"], n)[:, None] * event["amplitude"] * local_sd * SIGNATURES[signature]
    out[onset:onset + n] += bump[: max(0, min(n, len(X) - onset))]
    return out


def inject_fault(X: np.ndarray, onset: int, kind: str, local_sd: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, int, list[int]]:
    """Inject one fault; return (series, duration, affected channels).

    A fault that would run past the end of the series is truncated there."""
    out = X.copy()
    room = len(X) - onset

    def length(lo: int, hi: int) -> int:
        return min(int(rng.integers(lo, hi)), room)

    if kind == "dropout":
        ch = [int(rng.integers(4))]
        n = length(2, 13)
        out[onset:onset + n, ch] = 0.0
    elif kind == "drift":
        ch = [int(rng.integers(4))]
        n = length(8, 41)
        total = rng.uniform(2.0, 6.0) * rng.choice([-1, 1])
        out[onset:onset + n, ch[0]] += np.linspace(0, 1, n) * total * local_sd[ch[0]]
    elif kind == "noise_burst":
        ch = [int(rng.integers(4))]
        n = length(2, 13)
        out[onset:onset + n, ch[0]] += rng.normal(0, rng.uniform(3, 6), n) * local_sd[ch[0]]
    elif kind == "collapse":
        ch = [1]
        n = length(1, 7)
        out[onset:onset + n, 1] = X[onset:onset + n, 1] * rng.uniform(0.01, 0.05)
    elif kind == "fouling":
        # Gradual biofilm growth on the optical and conductivity probes:
        # turbidity drifts up while conductance drifts down, together.
        ch = [1, 3]
        n = length(12, 49)
        ramp = np.linspace(0, 1, n)
        out[onset:onset + n, 3] += ramp * rng.uniform(2.0, 5.0) * local_sd[3]
        out[onset:onset + n, 1] -= ramp * rng.uniform(2.0, 5.0) * local_sd[1]
    else:
        raise ValueError(kind)
    return out, n, ch
