"""Small, fast versions of the three experiments: determinism and sanity."""
import numpy as np
import pandas as pd
import pytest

from wqcadence import pillar1 as p1
from wqcadence import pillar2 as p2
from wqcadence.data import quiet_mask

pytestmark = pytest.mark.slow


def test_injection_trials_are_deterministic(data):
    a = p1.injection_trials(data, 15, seed=99)
    b = p1.injection_trials(data, 15, seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_candidate_onsets_are_quiet(data):
    for k in (1, 2):
        q = quiet_mask(data.deployment(k))
        for t in p1.candidate_onsets(q)[::25]:
            assert q[t:t + p1.MAX_DUR + p1.GRACE + 1].all()


def test_severe_events_are_mostly_detected_by_continuous(data):
    df = p1.injection_trials(data, 60, seed=5)
    sev = df[df.severity == "severe"]
    assert sev["Continuous-15|detected"].mean() >= 0.8


def test_background_has_one_row_per_phase(data):
    bg = p1.background(data)
    assert len(bg) == 2 * (2 + 2 + 1)      # deployments x phases (fixed 2, adaptive 2, continuous 1)


def test_pool_is_balanced_and_deterministic(data):
    a = p2.build_pool(data, 20, seed=4)
    b = p2.build_pool(data, 20, seed=4)
    assert sum(p["label"] for p in a) == 20
    np.testing.assert_array_equal(p2.featurise(a, "F4"), p2.featurise(b, "F4"))


def test_real_windows(data):
    real = p2.real_windows(data)
    assert sum(r["kind"] == "real_fault" for r in real) == 3
    assert sum(r["kind"] == "real_event" for r in real) == 4
