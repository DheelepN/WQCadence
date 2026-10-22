"""Sampling policies: fixed-interval, tiered adaptive, and continuous.

Every policy uses the same detector and the same alarm rule, so differences
between them come only from when they sample:

* alarm is raised when one sample scores >= tau2, or two consecutive samples
  score >= tau1 (a persistence rule, standard in event-detection software);
* alarm is cleared by the first sample scoring < tau1;
* the adaptive policy samples at ``base`` steps while in the NORMAL tier and at
  ``alert`` steps once a sample scores >= tau1, returning to NORMAL after
  ``hold`` consecutive samples below tau1.

Alarm state is forward-filled on the native 15-minute timeline until the next
sample: a consumer of a policy's feed holds the last belief until new data
arrive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import GapAwareEWMA, score

TAU1 = 1.8
TAU2 = 3.2


@dataclass(frozen=True)
class Policy:
    name: str
    base: int            # sampling interval in NORMAL tier (native steps)
    alert: int           # sampling interval once escalated (native steps)
    tau1: float = TAU1
    tau2: float = TAU2
    hold: int = 3
    persist: int = 2
    aggregate: str = "mean"

    @property
    def adaptive(self) -> bool:
        return self.alert < self.base


def fixed(interval: int, **kw) -> Policy:
    return Policy(f"Fixed-{interval * 15}", interval, interval, **kw)


def adaptive(base: int = 2, alert: int = 1, **kw) -> Policy:
    return Policy(f"Adaptive-{base * 15}/{alert * 15}", base, alert, **kw)


FIXED_30 = fixed(2)
CONTINUOUS = Policy("Continuous-15", 1, 1)
ADAPTIVE = adaptive(2, 1)


@dataclass
class PolicyRun:
    sample_idx: np.ndarray       # native indices actually sampled
    scores: np.ndarray           # score at each sample
    alarm: np.ndarray            # bool, native timeline (forward-filled)
    escalated: np.ndarray        # bool, native timeline: adaptive tier >= ALERT
    critical: np.ndarray         # bool, native timeline: last sample >= tau2

    @property
    def n_cycles(self) -> int:
        return int(self.sample_idx.size)


def run_policy(X: np.ndarray, policy: Policy, phase: int = 0,
               half_life_steps: float = 6.0) -> PolicyRun:
    """Run a policy over a model-space series X (T x C)."""
    T = X.shape[0]
    det = GapAwareEWMA(X.shape[1], half_life_steps)
    alarm = np.zeros(T, bool)
    escalated = np.zeros(T, bool)
    critical = np.zeros(T, bool)
    idx, scores = [], []

    t, last = phase, None
    above_run, below_run = 0, 0
    in_alarm, tier_alert = False, False
    while t < T:
        z = det.update(X[t], 1 if last is None else t - last)
        s = score(z, policy.aggregate) if last is not None else 0.0
        idx.append(t)
        scores.append(s)

        if s >= policy.tau1:
            above_run += 1
            below_run = 0
        else:
            above_run = 0
            below_run += 1

        if s >= policy.tau2 or above_run >= policy.persist:
            in_alarm = True
        elif s < policy.tau1:
            in_alarm = False

        if policy.adaptive:
            if s >= policy.tau1:
                tier_alert = True
            elif below_run >= policy.hold:
                tier_alert = False

        step = policy.alert if tier_alert else policy.base
        nxt = min(t + step, T)
        alarm[t:nxt] = in_alarm
        escalated[t:nxt] = tier_alert
        critical[t:nxt] = s >= policy.tau2
        last, t = t, nxt

    return PolicyRun(np.asarray(idx), np.asarray(scores), alarm, escalated, critical)


def alarm_episodes(alarm: np.ndarray, mask: np.ndarray | None = None) -> int:
    """Number of distinct alarm onsets, optionally only onsets where mask is True."""
    a = np.asarray(alarm, bool)
    onsets = a & ~np.r_[False, a[:-1]]
    if mask is not None:
        onsets &= np.asarray(mask, bool)
    return int(onsets.sum())
