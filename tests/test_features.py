import numpy as np

from aquasentinel.features import DECISION_STEPS, F4_NAMES, F6_NAMES, features_f4, features_f6
from aquasentinel.injection import SIGNATURES

rng = np.random.default_rng(0)
B = np.tile([10.0, 400.0, 6.0, 1.0], (20, 1)) + rng.normal(0, [0.1, 3, 0.03, 0.05], (20, 4))
SD = np.array([0.1, 3.0, 0.03, 0.05])


def event(sig="discharge", amp=8.0, n=6):
    return B[-1] + amp * SD * SIGNATURES[sig] * np.ones((n, 1))


def test_shapes():
    W = event()
    assert features_f4(W, B).shape == (len(F4_NAMES),)
    assert features_f6(W, B).shape == (len(F6_NAMES),)


def test_discharge_event_features():
    f = features_f4(event(), B)
    assert f[0] >= 3          # several channels deviate
    assert f[1] == 1          # turbidity/conductance same way, pH opposite
    assert f[2] == 1          # conductance plausible


def test_runoff_event_fails_original_directional_feature():
    """The a priori directional feature encodes a discharge signature; a
    dilution-type runoff event violates it."""
    assert features_f4(event("runoff"), B)[1] == 0


def test_conductance_collapse_is_implausible():
    W = np.tile(B[-1], (3, 1))
    W[:, 1] = 8.0
    assert features_f4(W, B)[2] == 0
    assert features_f6(W, B)[3] == 0


def test_zero_dropout_feature():
    W = np.tile(B[-1], (4, 1))
    W[:, 2] = 0.0
    assert features_f6(W, B)[5] == 0.25


def test_single_channel_fault_has_no_corroboration():
    W = np.tile(B[-1], (5, 1))
    W[:, 3] += 20 * SD[3]
    f = features_f6(W, B)
    assert f[0] == 1 and f[2] == 0 and f[1] < 3


def test_decision_window_truncation():
    W = event(n=DECISION_STEPS)
    long = np.vstack([W, np.zeros((50, 4))])
    np.testing.assert_array_equal(features_f6(W, B), features_f6(long, B))
