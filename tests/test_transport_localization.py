import numpy as np
import pytest

from aquasentinel import pillar3 as p3
from aquasentinel.transport import AREA, concentration


def test_zero_before_release():
    assert concentration(1000.0, 50.0, 500.0, 100.0, 0.3, 2.0, 1e4) == 0


def test_mass_is_conserved():
    x = np.linspace(-5000, 20000, 200_001)
    c = concentration(x, 7200.0, 1000.0, 0.0, 0.3, 4.0, 2.5e4)
    assert np.trapezoid(c, x) * AREA == pytest.approx(2.5e4, rel=1e-4)


def test_plume_centre_and_spread():
    x = np.linspace(0, 20000, 200_001)
    tau, v, D = 3600.0, 0.4, 5.0
    c = concentration(x, tau, 1000.0, 0.0, v, D, 1e4)
    w = c / c.sum()
    mean = (w * x).sum()
    var = (w * (x - mean) ** 2).sum()
    assert mean == pytest.approx(1000 + v * tau, abs=1)
    assert var == pytest.approx(2 * D * tau, rel=1e-3)


def test_fixed_cadence_sampling_counts():
    row = np.zeros(577)
    assert p3.sample(row, p3.Cadence("F30", 6, 6), 0).size == 97
    assert p3.sample(row, p3.Cadence("F5", 1, 1), 0).size == 577


def test_adaptive_cadence_escalates_on_detection():
    row = np.zeros(577)
    row[120:180] = 1.0
    idx = p3.sample(row, p3.Cadence("A", 6, 1), 0)
    inside = idx[(idx >= 120) & (idx < 180)]
    assert np.all(np.diff(inside) == 1)
    assert idx[-1] - idx[-2] == 6


def test_inversion_recovers_source_without_knowing_release_time(monkeypatch):
    monkeypatch.setattr(p3, "NOISE_REL", 0.001)
    monkeypatch.setattr(p3, "NOISE_ABS", 0.0005)
    rng = np.random.default_rng(11)
    errs = []
    for s in range(4):
        sc = p3.draw_scenario(rng, n_stations=5)
        r = p3.run_scenario(sc, s, cadences=(p3.CADENCES[-1],), estimators=("inversion",))[0]
        if not np.isnan(r["inversion_error"]):
            errs.append(r["inversion_error"])
    assert errs and max(errs) < 50


def test_no_localization_with_fewer_than_two_detections():
    assert p3.estimate_oracle({1000.0: 5.0}, 0.0) is None
