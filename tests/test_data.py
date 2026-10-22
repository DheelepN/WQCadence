import numpy as np
import pandas as pd

from aquasentinel.data import event_runs, model_matrix, quiet_mask, to_model_space


def test_record_counts_match_source_cleaned_files(data):
    s = data.series
    assert len(s) == 2775
    assert s.groupby("deployment").size().to_dict() == {1: 1455, 2: 1320}


def test_no_missing_values(data):
    assert not data.series.isna().any().any()


def test_regular_15_minute_spacing_within_deployments(data):
    for k in (1, 2):
        dt = data.deployment(k)["datetime"].diff().dropna()
        assert (dt.dt.total_seconds().between(15 * 60 - 2, 15 * 60 + 2)).all()


def test_source_labels_merged_completely(data):
    s = data.series
    assert s["event_sct"].sum() == 275
    assert s["event_verified"].sum() == 940
    assert len(event_runs(s.loc[s.deployment == 1, "event_verified"].to_numpy())) == 4


def test_real_fault_windows(data):
    assert len(data.faults) == 3
    assert sum(len(f.window) for f in data.faults) == 11
    for f in data.faults:
        assert f.baseline.shape == (20, 4)
        assert (f.window[:, 1] < 25).all()          # implausible conductance
        assert (f.baseline[:, 1] > 200).all()       # baseline is in-river water


def test_quiet_mask_excludes_flags_with_buffer(data):
    df = data.deployment(1)
    q = quiet_mask(df, buffer_steps=4)
    flagged = (df.event_sct == 1) | (df.event_verified == 1)
    assert not q[flagged.to_numpy()].any()
    s, _ = event_runs(flagged.to_numpy())[1]
    assert not q[s - 4:s].any()


def test_model_space_transforms_only_turbidity():
    raw = np.array([[10.0, 400.0, 6.0, 0.0], [11.0, 410.0, 6.1, np.e - 1]])
    m = to_model_space(raw)
    np.testing.assert_allclose(m[:, :3], raw[:, :3])
    np.testing.assert_allclose(m[:, 3], [0.0, 1.0])


def test_model_matrix_shape(data):
    assert model_matrix(data.deployment(2)).shape == (1320, 4)
