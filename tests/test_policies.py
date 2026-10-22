import numpy as np

from wqcadence.detector import SD_FLOOR
from wqcadence.policies import ADAPTIVE, CONTINUOUS, FIXED_30, alarm_episodes, fixed, run_policy

FLAT = np.tile([10.0, 400.0, 6.0, 1.0], (200, 1))


def bump(series, t, z):
    out = series.copy()
    out[t] += z * SD_FLOOR
    return out


def test_sampling_counts():
    assert run_policy(FLAT, CONTINUOUS).n_cycles == 200
    assert run_policy(FLAT, FIXED_30).n_cycles == 100
    assert run_policy(FLAT, FIXED_30, phase=1).n_cycles == 100
    assert run_policy(FLAT, fixed(4)).n_cycles == 50


def test_adaptive_equals_fixed_when_nothing_happens():
    a, f = run_policy(FLAT, ADAPTIVE), run_policy(FLAT, FIXED_30)
    np.testing.assert_array_equal(a.sample_idx, f.sample_idx)
    assert not a.alarm.any() and not a.escalated.any()


def test_adaptive_escalates_after_alert_score():
    X = FLAT.copy()
    X[100:] += 2.5 * SD_FLOOR       # sustained shift, score 2.5 on first sample
    run = run_policy(X, ADAPTIVE)
    i = np.searchsorted(run.sample_idx, 100)
    assert run.sample_idx[i] == 100
    assert run.sample_idx[i + 1] - run.sample_idx[i] == 1


def test_persistence_rule():
    one = run_policy(bump(FLAT, 100, 2.5), CONTINUOUS)          # tau1 <= s < tau2, once
    assert not one.alarm.any()
    big = run_policy(bump(FLAT, 100, 5.0), CONTINUOUS)          # s >= tau2
    assert big.alarm[100]
    X = bump(bump(FLAT, 100, 2.5), 101, 2.5)
    two = run_policy(X, CONTINUOUS)
    assert two.alarm[101] and not two.alarm[100]


def test_alarm_is_forward_filled_until_next_sample():
    run = run_policy(bump(FLAT, 100, 5.0), FIXED_30)
    assert run.alarm[100] and run.alarm[101] and not run.alarm[102]


def test_alarm_episodes():
    a = np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)
    assert alarm_episodes(a) == 3
    mask = np.array([1, 1, 1, 1, 0, 1, 1, 1], bool)
    assert alarm_episodes(a, mask) == 2


def test_stale_alarm_is_not_credited_as_detection():
    from wqcadence.pillar1 import _detect
    X = bump(FLAT, 100, 5.0)                      # false alarm at 100, held until 102
    run = run_policy(X, FIXED_30)
    assert run.alarm[101]
    hit, _ = _detect(run, onset=101, dur=1)       # window 101..105 contains samples 102, 104
    assert not hit
