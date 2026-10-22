"""Pillar 1: adaptive vs fixed-interval vs continuous sampling.

Two separate evaluations, because they answer different questions:

* Background false alarms are measured once per policy on the real,
  uninjected series (averaged over sampling phase). Repeating an injection
  trial does not create new background, so per-trial false-alarm statistics
  would be pseudo-replicated.
* Detection recall and latency are measured by Monte Carlo injection of
  synthetic events into real quiet periods, with every policy evaluated on the
  identical injected window (paired design).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import STEP_MINUTES
from .data import OwenmoreData, model_matrix, quiet_mask
from .injection import SEVERITY, draw_event, inject_event, robust_sd
from .policies import ADAPTIVE, CONTINUOUS, FIXED_30, Policy, adaptive, alarm_episodes, fixed, run_policy

WARMUP = 192          # 2 days of detector warm-up before each injected onset
SD_WINDOW = 96        # 1 day of preceding data for the local robust SD
GRACE = 4             # 1 h after event end still counts as a detection
MAIN = (FIXED_30, ADAPTIVE, CONTINUOUS)
SWEEP = tuple(fixed(k) for k in (1, 2, 3, 4, 6, 8)) + tuple(adaptive(k, 1) for k in (2, 3, 4, 6, 8))
MAX_DUR = max(v[3] for v in SEVERITY.values())


def candidate_onsets(quiet: np.ndarray) -> np.ndarray:
    """Onsets where the event window plus grace (and 2 h before) is quiet."""
    n = len(quiet)
    ok = []
    for t in range(WARMUP, n - MAX_DUR - GRACE):
        if quiet[t - 8:t + MAX_DUR + GRACE + 1].all() and quiet[t - SD_WINDOW:t].sum() >= 24:
            ok.append(t)
    return np.asarray(ok)


def _detect(run, onset: int, dur: int) -> tuple[bool, float]:
    """First sample taken at or after onset (and before end + grace) whose
    alarm state is on. An alarm already showing at onset from earlier samples
    is not credited: it carries no information about the event."""
    idx = run.sample_idx
    idx = idx[(idx >= onset) & (idx < onset + dur + GRACE)]
    hit = idx[run.alarm[idx]]
    if hit.size == 0:
        return False, np.nan
    return True, float((hit[0] - onset) * STEP_MINUTES)


def injection_trials(data: OwenmoreData, n_trials: int, seed: int,
                     policies: tuple[Policy, ...] = MAIN) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    deps = {k: (model_matrix(data.deployment(k)), quiet_mask(data.deployment(k))) for k in (1, 2)}
    onsets = {k: candidate_onsets(q) for k, (_, q) in deps.items()}
    weights = np.array([len(onsets[1]), len(onsets[2])], float)
    rows = []
    for trial in range(n_trials):
        k = int(rng.choice([1, 2], p=weights / weights.sum()))
        X, q = deps[k]
        onset = int(rng.choice(onsets[k]))
        ev = draw_event(rng, severity=list(SEVERITY)[trial % 3])
        pre = X[onset - SD_WINDOW:onset][q[onset - SD_WINDOW:onset]]
        lo, hi = onset - WARMUP, onset + ev["duration"] + GRACE + 1
        Xi = inject_event(X, onset, ev, robust_sd(pre))[lo:hi]
        phase = int(rng.integers(0, 2))
        rec = {"trial": trial, "deployment": k, "onset": onset, **ev, "phase": phase}
        for p in policies:
            run = run_policy(Xi, p, phase=phase % p.base)
            hit, lat = _detect(run, WARMUP, ev["duration"])
            rec[f"{p.name}|detected"] = hit
            rec[f"{p.name}|latency"] = lat
            rec[f"{p.name}|cycles"] = int(np.sum(run.sample_idx >= WARMUP))
            rec[f"{p.name}|critical"] = bool(run.critical[WARMUP:WARMUP + ev["duration"] + GRACE].any())
        rows.append(rec)
    return pd.DataFrame(rows)


def background(data: OwenmoreData, policies: tuple[Policy, ...] = MAIN,
               warmup: int = 96) -> pd.DataFrame:
    """False alarms and cost on the real uninjected series, averaged over phase."""
    rows = []
    for p in policies:
        for k in (1, 2):
            df = data.deployment(k)
            X, q = model_matrix(df), quiet_mask(df)
            q_eval = q.copy()
            q_eval[:warmup] = False
            for phase in range(p.base):
                run = run_policy(X, p, phase=phase)
                rows.append({
                    "policy": p.name, "persist": p.persist, "tau1": p.tau1, "deployment": k, "phase": phase,
                    "quiet_steps": int(q_eval.sum()),
                    "alarm_quiet_steps": int(run.alarm[q_eval].sum()),
                    "false_episodes": alarm_episodes(run.alarm, q_eval),
                    "cycles": run.n_cycles, "days": len(df) / 96.0,
                    "flagged_steps": int((~q).sum()),
                    "alarm_flagged_steps": int(run.alarm[~q].sum()),
                })
    return pd.DataFrame(rows)


def summarise_background(bg: pd.DataFrame) -> pd.DataFrame:
    g = bg.groupby(["policy", "persist", "tau1", "deployment"]).mean(numeric_only=True).reset_index()
    tot = g.groupby(["policy", "persist", "tau1"]).sum(numeric_only=True).reset_index()
    tot["far_per_1000"] = 1000 * tot["alarm_quiet_steps"] / tot["quiet_steps"]
    tot["false_episodes_per_day"] = tot["false_episodes"] / (tot["quiet_steps"] / 96)
    tot["cycles_per_day"] = tot["cycles"] / tot["days"]
    tot["real_event_coverage"] = tot["alarm_flagged_steps"] / tot["flagged_steps"]
    return tot.drop(columns=["deployment", "phase"])


def real_event_detection(data: OwenmoreData, policies: tuple[Policy, ...] = MAIN,
                         before: int = 4, after: int = 8) -> pd.DataFrame:
    """Does each policy alarm near the onset of each expert-verified real event?

    Detection window: from 1 h before to 2 h after the verified onset (the
    source study's labels are manual, so onsets are approximate). Each policy
    is run on the full uninjected deployment and averaged over sampling phase."""
    from .data import event_runs
    rows = []
    for k in (1, 2):
        df = data.deployment(k)
        X = model_matrix(df)
        runs = event_runs(df["event_verified"].to_numpy())
        for p in policies:
            for phase in range(p.base):
                run = run_policy(X, p, phase=phase)
                for i, (s, e) in enumerate(runs):
                    lo = max(0, s - before)
                    hit = np.flatnonzero(run.alarm[lo:s + after + 1])
                    rows.append({"policy": p.name, "aggregate": p.aggregate, "deployment": k,
                                 "event": f"{df['datetime'][s]:%Y-%m-%d %H:%M}",
                                 "duration_h": (e - s + 1) * STEP_MINUTES / 60, "phase": phase,
                                 "detected": hit.size > 0,
                                 "latency_min": float((lo + hit[0] - s) * STEP_MINUTES) if hit.size else np.nan})
    return pd.DataFrame(rows)


def with_aggregate(policies, aggregate: str):
    return tuple(replace(p, aggregate=aggregate, name=f"{p.name} [{aggregate}]") for p in policies)


def with_persist(policies, persist: int):
    return tuple(replace(p, persist=persist) for p in policies)


def with_tau1(policies, tau1: float):
    return tuple(replace(p, tau1=tau1, name=f"{p.name}@{tau1:g}") for p in policies)
