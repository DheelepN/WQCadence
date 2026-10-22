import numpy as np
import pytest

from aquasentinel.detector import SD_FLOOR, Z_CLIP, GapAwareEWMA, score


def test_first_update_returns_zero_and_initialises():
    d = GapAwareEWMA()
    z = d.update(np.array([10.0, 400.0, 6.0, 1.0]))
    assert np.all(z == 0)
    np.testing.assert_allclose(d.mean, [10.0, 400.0, 6.0, 1.0])


def test_constant_input_scores_zero():
    d = GapAwareEWMA()
    x = np.array([10.0, 400.0, 6.0, 1.0])
    for _ in range(20):
        z = d.update(x)
    assert np.all(z == 0)


def test_scores_against_pre_update_baseline():
    d = GapAwareEWMA()
    base = np.array([10.0, 400.0, 6.0, 1.0])
    for _ in range(10):
        d.update(base)
    z = d.update(base + SD_FLOOR * 2)
    np.testing.assert_allclose(z, 2.0)


def test_decay_is_wall_clock_not_per_cycle():
    """After a gap equal to the half-life, half the weight stays on the old baseline."""
    d = GapAwareEWMA(half_life_steps=6)
    d.update(np.zeros(4))
    d.update(np.ones(4), gap_steps=6)
    np.testing.assert_allclose(d.mean, 0.5)


def test_two_single_steps_equal_one_double_gap_for_mean_weighting():
    a, b = GapAwareEWMA(), GapAwareEWMA()
    a.update(np.zeros(4)); b.update(np.zeros(4))
    a.update(np.ones(4), gap_steps=2)
    r = b.r
    np.testing.assert_allclose(a.mean, 1 - r ** 2)


def test_z_is_clipped():
    d = GapAwareEWMA()
    d.update(np.zeros(4))
    z = d.update(np.full(4, 1e6))
    assert np.all(np.abs(z) == Z_CLIP)


def test_score_aggregations():
    z = np.array([10.0, 0.0, 0.0, 0.0])
    assert score(z, "mean") == pytest.approx(2.5)
    assert score(z, "rms") == pytest.approx(5.0)
    with pytest.raises(ValueError):
        score(z, "max")


@pytest.mark.parametrize("gap", [1, 2, 4, 8])
def test_z_is_calibrated_at_every_cadence(gap):
    """For stationary noise, E[z^2] stays near 1 whatever the sampling gap.

    It rises slightly at long gaps (fewer samples inside the 6 h variance
    window, so z is t-like); the naive EW variance gives ~5.4 at gap 8."""
    rng = np.random.default_rng(gap)
    x = rng.normal(0, 1.0, (20_000, 4)) * np.array([1, 50, 0.5, 1])
    d = GapAwareEWMA(sd_floor=np.full(4, 1e-6))
    zs = [d.update(row, gap_steps=gap) for row in x[::gap]]
    z2 = np.mean(np.square(zs[200:]))
    assert 0.9 < z2 < 1.3
