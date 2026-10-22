import numpy as np
import pytest

from wqcadence.stats import bootstrap_ci, holm, mcnemar_exact, wilcoxon_paired, wilson


def test_mcnemar_exact_known_value():
    a = np.r_[np.ones(10, bool), np.ones(5, bool)]
    b = np.r_[np.zeros(10, bool), np.ones(5, bool)]
    r = mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (10, 0)
    assert r["p"] == pytest.approx(2 * 0.5 ** 10)


def test_mcnemar_no_discordance():
    x = np.array([1, 0, 1], bool)
    assert mcnemar_exact(x, x)["p"] == 1.0


def test_wilson_interval():
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-3)


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    x = np.random.default_rng(0).normal(5, 1, 200)
    a, b = bootstrap_ci(x, seed=1), bootstrap_ci(x, seed=1)
    assert a == b and a[0] < x.mean() < a[1]


def test_wilcoxon_all_ties():
    r = wilcoxon_paired(np.ones(5), np.ones(5))
    assert r["p"] == 1.0 and r["n_nonzero"] == 0


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
