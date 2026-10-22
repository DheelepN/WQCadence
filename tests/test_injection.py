import numpy as np
import pytest

from wqcadence.detector import SD_FLOOR
from wqcadence.injection import (FAULT_TYPES, SEVERITY, SHAPES, SIGNATURES, draw_event, inject_event,
                                    inject_fault, profile, robust_sd)

BASE = np.tile([10.0, 400.0, 6.0, 1.0], (100, 1))
SD = np.array([0.2, 5.0, 0.05, 0.1])


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_profile_unit_peak(shape, n):
    p = profile(shape, n)
    assert p.shape == (n,)
    assert p.max() == pytest.approx(1.0)
    assert (p >= 0).all()


def test_draw_event_respects_severity_bands():
    rng = np.random.default_rng(0)
    for _ in range(300):
        ev = draw_event(rng)
        a_lo, a_hi, d_lo, d_hi = SEVERITY[ev["severity"]]
        assert a_lo <= ev["amplitude"] <= a_hi
        assert d_lo <= ev["duration"] <= d_hi


@pytest.mark.parametrize("sig", list(SIGNATURES))
def test_inject_event_is_local_and_signed(sig):
    ev = {"severity": "severe", "shape": "ramp_plateau", "amplitude": 4.0, "duration": 10}
    out = inject_event(BASE, 50, ev, SD, signature=sig)
    np.testing.assert_array_equal(out[:50], BASE[:50])
    np.testing.assert_array_equal(out[60:], BASE[60:])
    peak = (out - BASE)[50:60]
    i = np.argmax(np.abs(peak), axis=0)
    signs = np.sign(peak[i, range(4)])
    np.testing.assert_array_equal(signs, np.sign(SIGNATURES[sig]))
    np.testing.assert_allclose(np.abs(peak).max(axis=0), 4.0 * SD * np.abs(SIGNATURES[sig]))


@pytest.mark.parametrize("kind", FAULT_TYPES)
def test_inject_fault_affects_declared_channels_only(kind):
    rng = np.random.default_rng(3)
    out, n, ch = inject_fault(BASE, 30, kind, SD, rng)
    changed = np.flatnonzero(np.any(out != BASE, axis=0))
    assert set(changed) <= set(ch)
    assert np.all(out[30 + n:] == BASE[30 + n:])


def test_collapse_is_physically_implausible():
    out, n, _ = inject_fault(BASE, 30, "collapse", SD, np.random.default_rng(1))
    assert (out[30:30 + n, 1] <= 0.05 * 400).all()


def test_fouling_moves_turbidity_and_conductance_oppositely():
    out, n, ch = inject_fault(BASE, 30, "fouling", SD, np.random.default_rng(2))
    d = out[30 + n - 1] - BASE[0]
    assert ch == [1, 3] and d[3] > 0 and d[1] < 0


def test_robust_sd_floor():
    np.testing.assert_allclose(robust_sd(BASE), SD_FLOOR)


@pytest.mark.parametrize("kind", FAULT_TYPES)
def test_fault_near_series_end_is_truncated(kind):
    for seed in range(20):
        out, n, _ = inject_fault(BASE, 95, kind, SD, np.random.default_rng(seed))
        assert out.shape == BASE.shape and 1 <= n <= 5
