"""Statistical helpers: exact McNemar, Wilson intervals, bootstrap, Holm."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test on paired binary outcomes."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int(np.sum(a & ~b))   # a yes, b no
    n10 = int(np.sum(~a & b))   # a no, b yes
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(n01, n, 0.5).pvalue)
    return {"a_only": n01, "b_only": n10, "p": p}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 5000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, (n_boot, x.size))
    boots = np.apply_along_axis(stat, 1, x[idx])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return (float(lo), float(hi))


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    if nz.size == 0:
        return {"n": int(d.size), "n_nonzero": 0, "statistic": float("nan"), "p": 1.0,
                "median_diff": 0.0, "mean_diff": 0.0}
    res = stats.wilcoxon(nz)
    return {"n": int(d.size), "n_nonzero": int(nz.size), "statistic": float(res.statistic),
            "p": float(res.pvalue), "median_diff": float(np.median(d)), "mean_diff": float(np.mean(d))}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out
