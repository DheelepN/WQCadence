"""Pillar 2: discriminating sensor faults from water-quality events."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import OwenmoreData, event_runs, model_matrix, quiet_mask
from .features import DECISION_STEPS, FEATURE_SETS
from .injection import draw_event, inject_event, inject_fault, robust_sd
from .pillar1 import SD_WINDOW, candidate_onsets

BASELINE = 20
TRAIN_FAULTS = ("dropout", "drift", "noise_burst", "collapse")


def build_pool(data: OwenmoreData, n_per_class: int, seed: int,
               event_signatures=("discharge",), fault_types=TRAIN_FAULTS) -> list[dict]:
    """Labelled synthetic windows (label 1 = fault, 0 = event)."""
    rng = np.random.default_rng(seed)
    deps = {k: (model_matrix(data.deployment(k)), quiet_mask(data.deployment(k))) for k in (1, 2)}
    onsets = {k: candidate_onsets(q) for k, (_, q) in deps.items()}
    w = np.array([len(onsets[1]), len(onsets[2])], float)

    def site():
        k = int(rng.choice([1, 2], p=w / w.sum()))
        X, q = deps[k]
        t = int(rng.choice(onsets[k]))
        return X, t, robust_sd(X[t - SD_WINDOW:t][q[t - SD_WINDOW:t]])

    pool = []
    for i in range(n_per_class):
        X, t, sd = site()
        ev = draw_event(rng)
        sig = event_signatures[i % len(event_signatures)]
        Xi = inject_event(X, t, ev, sd, signature=sig)
        pool.append({"label": 0, "kind": f"event_{sig}", "severity": ev["severity"],
                     "W": Xi[t:t + max(ev["duration"], 1)], "B": Xi[t - BASELINE:t]})
    for i in range(n_per_class):
        X, t, sd = site()
        kind = fault_types[i % len(fault_types)]
        Xi, n, _ = inject_fault(X, t, kind, sd, rng)
        pool.append({"label": 1, "kind": kind, "severity": "",
                     "W": Xi[t:t + n], "B": Xi[t - BASELINE:t]})
    return pool


def featurise(pool: list[dict], fset: str) -> np.ndarray:
    fn, _ = FEATURE_SETS[fset]
    return np.array([fn(p["W"], p["B"]) for p in pool])


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))


def random_forest(seed: int = 0):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed)


def cross_validate(X: np.ndarray, y: np.ndarray, make_model, seed: int,
                   n_splits: int = 5, n_repeats: int = 10) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    f1s, ps, rs = [], [], []
    cm = np.zeros((2, 2), int)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        m = make_model().fit(X[tr], y[tr])
        pred = m.predict(X[te])
        f1s.append(f1_score(y[te], pred))
        ps.append(precision_score(y[te], pred, zero_division=0))
        rs.append(recall_score(y[te], pred))
        if i < n_splits:          # first repeat only, so the matrix sums to n
            cm += confusion_matrix(y[te], pred, labels=[0, 1])
    return {"f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s, ddof=1)),
            "precision_mean": float(np.mean(ps)), "recall_mean": float(np.mean(rs)),
            "confusion_first_repeat": cm.tolist()}


def rule_baseline(X: np.ndarray, y: np.ndarray) -> dict:
    """Fault if at most one channel deviates (column 0 is n_channels_deviant in both sets)."""
    pred = (X[:, 0] <= 1).astype(int)
    return {"f1": float(f1_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred))}


def leave_one_type_out(pool: list[dict], X: np.ndarray, make_model) -> dict:
    kinds = np.array([p["kind"] for p in pool])
    y = np.array([p["label"] for p in pool])
    out = {}
    for k in TRAIN_FAULTS:
        tr = kinds != k
        m = make_model().fit(X[tr], y[tr])
        out[k] = float(np.mean(m.predict(X[kinds == k]) == 1))
    return out


def real_windows(data: OwenmoreData) -> list[dict]:
    """Real external windows: implausible-record faults and expert-verified events."""
    out = [{"label": 1, "kind": "real_fault", "name": f"D{f.deployment} {f.start:%Y-%m-%d %H:%M}",
            "W": f.window, "B": f.baseline, "n": len(f.window)} for f in data.faults]
    for k in (1, 2):
        df = data.deployment(k)
        X = model_matrix(df)
        for s, e in event_runs(df["event_verified"].to_numpy()):
            if s < BASELINE:
                continue
            out.append({"label": 0, "kind": "real_event", "name": f"D{k} {df['datetime'][s]:%Y-%m-%d %H:%M}",
                        "W": X[s:e + 1], "B": X[s - BASELINE:s], "n": e - s + 1})
    return out


def external_check(train_pool: list[dict], real: list[dict], fset: str, make_model) -> pd.DataFrame:
    Xtr = featurise(train_pool, fset)
    ytr = np.array([p["label"] for p in train_pool])
    m = make_model().fit(Xtr, ytr)
    Xr = featurise(real, fset)
    prob = m.predict_proba(Xr)[:, 1]
    _, names = FEATURE_SETS[fset]
    rows = []
    for r, p, x in zip(real, prob, Xr):
        rows.append({"window": r["name"], "kind": r["kind"], "records": r["n"],
                     "true_label": "fault" if r["label"] else "event",
                     "p_fault": float(p), "predicted": "fault" if p >= 0.5 else "event",
                     "correct": (p >= 0.5) == bool(r["label"]),
                     **{n: float(v) for n, v in zip(names, x)}})
    return pd.DataFrame(rows)


def coefficients(train_pool: list[dict], fset: str) -> dict:
    X = featurise(train_pool, fset)
    y = np.array([p["label"] for p in train_pool])
    m = logistic().fit(X, y)
    _, names = FEATURE_SETS[fset]
    return dict(zip(names, m[-1].coef_[0].round(3).tolist()))


__all__ = ["build_pool", "featurise", "logistic", "random_forest", "cross_validate", "rule_baseline",
           "leave_one_type_out", "real_windows", "external_check", "coefficients", "DECISION_STEPS"]
