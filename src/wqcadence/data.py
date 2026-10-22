"""Loading and preparing the Owenmore River YSI sonde data (Cronin et al., 2025).

All processing happens in "model space": temperature (deg C), specific
conductance (uS/cm) and pH are used as recorded; turbidity is log1p-transformed
because it is heavy-tailed (0.3 to ~950 NTU in this dataset).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import CHANNELS

RAW_FILES = {1: "OW061024.txt", 2: "OW221024.txt"}
CLEAN_FILES = {1: "OW061024C.csv", 2: "OW221024C.csv"}
LABEL_FILE = "OW061024_051124C_ManVer_TurbEvent.csv"

# Records whose specific conductance is below this fraction of the deployment
# median are physically implausible for this river (near-distilled water).
IMPLAUSIBLE_SPCOND_FRACTION = 0.05


def to_model_space(raw: np.ndarray) -> np.ndarray:
    """Raw channel matrix (T x 4, CHANNELS order) -> model space."""
    out = np.asarray(raw, dtype=float).copy()
    out[..., 3] = np.log1p(np.clip(out[..., 3], 0.0, None))
    return out


def read_ysi(path: Path) -> pd.DataFrame:
    """Read a YSI EcoWatch export (4 header lines) into a tidy frame."""
    df = pd.read_csv(path, skiprows=[0, 2, 3, 4], skipinitialspace=True)
    names = [c.strip() for c in df.columns]
    # The export has two "pH" columns (pH units, then mV); keep the first.
    ph_col = df.columns[names.index("pH")]
    df.columns = [n if c != ph_col else "pH_units" for n, c in zip(names, df.columns)]
    out = pd.DataFrame({
        "datetime": pd.to_datetime(df["Date"].str.strip() + " " + df["Time"].str.strip(),
                                   format="%d/%m/%Y %H:%M:%S"),
        "temp": df["Temp"].astype(float),
        "spcond": df["SpCond"].astype(float),
        "ph": df["pH_units"].astype(float),
        "turbidity": df["Turbid+"].astype(float),
    })
    return out


@dataclass
class FaultWindow:
    """A run of physically implausible raw records with an adjacent in-river baseline."""
    deployment: int
    start: pd.Timestamp
    end: pd.Timestamp
    window: np.ndarray      # model-space values, n x 4
    baseline: np.ndarray    # model-space values, 20 x 4 (adjacent in-river records)
    baseline_side: str      # "before" or "after"


@dataclass
class OwenmoreData:
    series: pd.DataFrame            # in-river records, both deployments
    faults: list[FaultWindow]
    raw: dict[int, pd.DataFrame]

    def deployment(self, k: int) -> pd.DataFrame:
        return self.series[self.series["deployment"] == k].reset_index(drop=True)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of contiguous True runs."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def load_owenmore(raw_dir: Path | str, baseline_len: int = 20) -> OwenmoreData:
    raw_dir = Path(raw_dir)
    labels = pd.read_csv(raw_dir / LABEL_FILE)
    labels.columns = [c.strip() for c in labels.columns]
    labels["minute"] = pd.to_datetime(labels["DateTime"], format="%d/%m/%Y %H:%M")
    labels = labels[["minute", "Possible_Owenmore_Event_Alarm", "Verified_OW_Event_Alarm"]]

    frames, faults, raws = [], [], {}
    for k in (1, 2):
        raw = read_ysi(raw_dir / RAW_FILES[k])
        raws[k] = raw
        # The source study's cleaned file defines the in-river period.
        clean = pd.read_csv(raw_dir / CLEAN_FILES[k])
        clean_times = pd.to_datetime(clean["DateTime"])
        in_river = raw["datetime"].isin(set(clean_times))
        dep = raw[in_river].copy()
        dep["deployment"] = k
        dep["minute"] = dep["datetime"].dt.floor("min")
        dep = dep.merge(labels, on="minute", how="left", validate="one_to_one")
        dep = dep.rename(columns={"Possible_Owenmore_Event_Alarm": "event_sct",
                                  "Verified_OW_Event_Alarm": "event_verified"})
        dep = dep.drop(columns="minute")
        frames.append(dep)

        # Real fault windows: implausible conductance anywhere in the raw log.
        med = raw.loc[in_river, "spcond"].median()
        bad = (raw["spcond"] < IMPLAUSIBLE_SPCOND_FRACTION * med).to_numpy()
        river_idx = np.flatnonzero(in_river.to_numpy())
        model = to_model_space(raw[list(CHANNELS)].to_numpy())
        for s, e in _runs(bad):
            before = river_idx[river_idx < s][-baseline_len:]
            after = river_idx[river_idx > e][:baseline_len]
            if len(before) == baseline_len:
                base, side = model[before], "before"
            else:
                base, side = model[after], "after"
            faults.append(FaultWindow(k, raw["datetime"][s], raw["datetime"][e],
                                      model[s:e + 1], base, side))

    series = pd.concat(frames, ignore_index=True)
    series[["event_sct", "event_verified"]] = series[["event_sct", "event_verified"]].astype(int)
    return OwenmoreData(series=series, faults=faults, raw=raws)


def event_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index runs of a 0/1 event flag."""
    return _runs(np.asarray(flags).astype(bool))


def quiet_mask(df: pd.DataFrame, buffer_steps: int = 4) -> np.ndarray:
    """True where neither the source study's statistical flag nor its expert
    verification marks an event, with a buffer either side of every flagged run."""
    flagged = (df["event_sct"].to_numpy() == 1) | (df["event_verified"].to_numpy() == 1)
    n = len(flagged)
    out = ~flagged
    for s, e in _runs(flagged):
        out[max(0, s - buffer_steps):min(n, e + buffer_steps + 1)] = False
    return out


def model_matrix(df: pd.DataFrame) -> np.ndarray:
    return to_model_space(df[list(CHANNELS)].to_numpy())
