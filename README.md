# WQCadence — reproducibility bundle

Code, data provenance, tests, results and manuscript for:

> *Adaptive sampling, fault-aware anomaly discrimination and multi-station source
> localization for river water-quality surveillance: an empirical and simulation benchmark.*

Every number, table and figure in the manuscript is produced by the scripts below
from the public Owenmore River dataset (Cronin et al., 2025). Nothing in the
paper is typed in by hand: `scripts/build_paper.py` reads `results/summary.json`.

## Layout

| Path | Contents |
|---|---|
| `data/raw/` | Source files from the Cronin et al. (2025) repository (MIT licence), unmodified |
| `src/wqcadence/` | The framework: data loading, detector, sampling policies, injection, features, transport model, statistics, and the three experiments |
| `tests/` | `pytest` suite (unit tests + small end-to-end runs) |
| `scripts/run_experiments.py` | Runs all experiments → `results/` |
| `scripts/make_figures.py` | Builds all figures → `figures/` (PNG + PDF) |
| `scripts/build_paper.py` | Builds the manuscript → `paper/` |
| `results/` | Raw per-trial outputs (CSV) and `summary.json` |

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pytest                       # ~10 s
python scripts/run_experiments.py      # ~10 min on 8 cores
python scripts/make_figures.py
python scripts/build_paper.py
```

`python scripts/run_experiments.py --quick` runs a small version in under a minute.

Random seeds are fixed in `scripts/run_experiments.py` (`SEEDS`). Pillar 3 runs
in parallel via joblib; scenarios are drawn from one seeded stream before
dispatch, so results do not depend on the number of workers.

## Data provenance

Source: <https://github.com/LCroninATU/Workflow-for-identifying-water-quality-hydrodynamic-events-in-time-series-data>
(Cronin, Taylor, Briciu Burghina, Lucy & Regan, 2025, *MethodsX* 15:103538,
doi:10.1016/j.mex.2025.103538). Files used:

* `OW061024.txt`, `OW221024.txt` — raw YSI EXO exports, deployments 1 and 2
* `OW061024C.csv`, `OW221024C.csv` — the source study's cleaned in-river records (define the in-river period)
* `OW061024_051124C_ManVer_TurbEvent.csv` — statistical-control-threshold flags and expert-verified event flags
* `OW061024_051124C_NoOutRange_TurbEvent.csv` — statistical flags (kept for reference)

The full source archive downloaded on 2026-10-22 (`repo.zip`) is kept locally and not redistributed; download it from the URL above.

## What is and is not validated

The study validates a software decision layer (when to sample, whether an
anomaly is a fault, where a release entered) against one river's real
background data plus synthetic ground truth. It does not validate any physical
sensing hardware. See the manuscript's Limitations section.
