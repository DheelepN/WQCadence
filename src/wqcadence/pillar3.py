"""Pillar 3: multi-station source localization and its dependence on cadence.

Each scenario releases an unknown mass at an unknown position and an unknown
time into a 10 km reach. Stations sample the plume under a cadence policy
(with measurement noise shared across policies, so comparisons are paired),
and the source is estimated two ways:

* ``oracle``: arrival-time regression x_i = x0 + v (t_i - t0), with t_i the
  time of a station's highest sample. It needs the release time t0 to be
  known, which it never is in practice, so it is an optimistic reference, not
  a deployable estimator.
* ``inversion``: weighted nonlinear least-squares fit of the
  advection-dispersion solution to every sample from every station, with
  (x0, t0, v, D, M) all unknown. Breakthrough-curve spread carries the
  travel-time information that arrival times alone lack, so sampling cadence
  matters directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares

from .transport import concentration

REACH = 10_000.0          # m
GRID = 300.0              # native simulation step, s (5 min)
HORIZON = 48 * 3600.0     # s
C_DET = 0.05              # detection threshold, mg/L (10x the noise floor)
NOISE_REL, NOISE_ABS = 0.05, 0.005


@dataclass(frozen=True)
class Cadence:
    name: str
    base: int    # grid steps between samples in normal mode
    alert: int   # grid steps once a sample exceeds C_DET
    hold: int = 3


CADENCES = (
    Cadence("Fixed-60", 12, 12),
    Cadence("Fixed-30", 6, 6),
    Cadence("Adaptive-30/15", 6, 3),
    Cadence("Fixed-15", 3, 3),
    Cadence("Adaptive-30/5", 6, 1),
    Cadence("Fixed-5", 1, 1),
)


def draw_scenario(rng: np.random.Generator, n_stations: int | None = None) -> dict:
    n = int(rng.integers(3, 7)) if n_stations is None else n_stations
    return {
        "x0": rng.uniform(0, 6000), "t0": rng.uniform(0, 6 * 3600),
        "v": rng.uniform(0.08, 0.6), "D": rng.uniform(0.5, 8.0),
        "M": 10 ** rng.uniform(3, 5),
        "stations": np.sort(rng.uniform(0, REACH, n)),
    }


def observe(sc: dict, rng: np.random.Generator) -> np.ndarray:
    """Noisy concentration on the native grid, stations x time."""
    t = np.arange(0, HORIZON + GRID, GRID)
    C = concentration(sc["stations"][:, None], t[None, :], sc["x0"], sc["t0"], sc["v"], sc["D"], sc["M"])
    noisy = C * (1 + NOISE_REL * rng.standard_normal(C.shape)) + NOISE_ABS * rng.standard_normal(C.shape)
    return np.clip(noisy, 0, None)


def sample(obs_row: np.ndarray, cad: Cadence, phase: int) -> np.ndarray:
    """Grid indices a station samples under a cadence policy."""
    idx, t, below, alert = [], phase % cad.base, 0, False
    while t < obs_row.size:
        idx.append(t)
        if obs_row[t] > C_DET:
            alert, below = True, 0
        else:
            below += 1
            if below >= cad.hold:
                alert = False
        t += cad.alert if alert else cad.base
    return np.asarray(idx)


def estimate_oracle(arrivals: dict[float, float], t0: float) -> float | None:
    if len(arrivals) < 2:
        return None
    x = np.array(list(arrivals.keys()))
    tau = np.array(list(arrivals.values())) - t0
    tc = tau - tau.mean()
    if not np.any(tc):
        return None
    b = np.dot(tc, x - x.mean()) / np.dot(tc, tc)
    a = x.mean() - b * tau.mean()       # position at tau = 0, i.e. the source
    return float(np.clip(a, 0, REACH))


def estimate_inversion(xs, ts, cs, arrivals: dict[float, float], rng: np.random.Generator,
                       n_starts: int = 6) -> float | None:
    if len(arrivals) < 2:
        return None
    first_x = min(arrivals)
    first_t = arrivals[first_x]
    w = 1.0 / (NOISE_REL * np.maximum(cs, C_DET) + NOISE_ABS)

    def resid(p):
        x0, t0, lv, lD, lM = p
        return (concentration(xs, ts, x0, t0, np.exp(lv), np.exp(lD), np.exp(lM)) - cs) * w

    lo = np.array([0.0, -6 * 3600.0, np.log(0.02), np.log(0.05), np.log(1.0)])
    hi = np.array([first_x, first_t, np.log(3.0), np.log(100.0), np.log(1e8)])
    ax = np.array(list(arrivals.keys()))
    at = np.array(list(arrivals.values()))
    v_guess = np.clip(np.polyfit(at, ax, 1)[0], 0.03, 2.5) if np.ptp(at) > 0 else 0.3
    best = None
    for k in range(n_starts):
        x0g = rng.uniform(0, first_x)
        t0g = first_t - (first_x - x0g) / v_guess
        p0 = np.array([x0g, t0g, np.log(v_guess), np.log(rng.uniform(0.5, 10)), np.log(1e4)])
        p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), x_scale=[1000, 3600, 1, 1, 1], max_nfev=400)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return None if best is None else float(best.x[0])


def run_scenario(sc: dict, seed: int, cadences=CADENCES, estimators=("oracle", "inversion")) -> list[dict]:
    rng = np.random.default_rng(seed)
    obs = observe(sc, rng)
    phases = rng.integers(0, 12, sc["stations"].size)
    t_grid = np.arange(obs.shape[1]) * GRID
    rows = []
    for cad in cadences:
        xs, ts, cs, arrivals, n_samples = [], [], [], {}, 0
        for i, xst in enumerate(sc["stations"]):
            idx = sample(obs[i], cad, int(phases[i]))
            n_samples += idx.size
            xs.append(np.full(idx.size, xst))
            ts.append(t_grid[idx])
            cs.append(obs[i, idx])
            if np.any(obs[i, idx] > C_DET):
                arrivals[float(xst)] = float(t_grid[idx[np.argmax(obs[i, idx])]])
        xs, ts, cs = map(np.concatenate, (xs, ts, cs))
        rec = {"cadence": cad.name, "n_stations": int(sc["stations"].size),
               "n_detecting": len(arrivals), "samples_per_station_day": n_samples / sc["stations"].size / 2.0,
               **{k: sc[k] for k in ("x0", "t0", "v", "D", "M")}}
        for est in estimators:
            if est == "oracle":
                xh = estimate_oracle(arrivals, sc["t0"])
            else:
                xh = estimate_inversion(xs, ts, cs, arrivals, np.random.default_rng(seed + 1))
            rec[f"{est}_x0"] = np.nan if xh is None else xh
            rec[f"{est}_error"] = np.nan if xh is None else abs(xh - sc["x0"])
        rows.append(rec)
    return rows


def _tag(rows: list[dict], s: int) -> list[dict]:
    for r in rows:
        r["scenario"] = s
    return rows


def run(n_scenarios: int, seed: int, cadences=CADENCES, n_stations: int | None = None,
        n_jobs: int = -1) -> pd.DataFrame:
    """Scenarios are drawn sequentially from one seeded stream, then evaluated
    in parallel; results do not depend on n_jobs."""
    rng = np.random.default_rng(seed)
    scenarios = [draw_scenario(rng, n_stations) for _ in range(n_scenarios)]
    out = Parallel(n_jobs=n_jobs)(
        delayed(run_scenario)(sc, seed * 100_000 + s, cadences) for s, sc in enumerate(scenarios))
    return pd.DataFrame([r for s, rows in enumerate(out) for r in _tag(rows, s)])


def run_station_count(n_scenarios: int, seed: int, counts=(2, 3, 4, 6, 8),
                      cadences=(CADENCES[1], CADENCES[2]), n_jobs: int = -1) -> pd.DataFrame:
    """Nested design: each scenario draws 8 stations and uses the first n of a
    random ordering, so station counts are compared on identical releases."""
    rng = np.random.default_rng(seed)
    jobs = []
    for s in range(n_scenarios):
        base = draw_scenario(rng, max(counts))
        order = rng.permutation(max(counts))
        for n in counts:
            jobs.append((s, dict(base, stations=np.sort(base["stations"][order[:n]]))))
    out = Parallel(n_jobs=n_jobs)(
        delayed(run_scenario)(sc, seed * 100_000 + s, cadences, ("inversion",)) for s, sc in jobs)
    return pd.DataFrame([r for (s, _), rows in zip(jobs, out) for r in _tag(rows, s)])
