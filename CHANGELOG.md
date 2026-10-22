# Changelog

## v1.0.0 (2026-10-22)

First public release. It accompanies the manuscript *"Where adaptive sampling pays off in river water-quality surveillance: detection, fault discrimination and source localization in a reproducible benchmark"* (Nunna, submitted).

### What WQCadence does

A Python framework that evaluates three decisions a river water-quality monitoring station has to make:

1. **When to sample.** Fixed-interval, tiered adaptive and continuous sampling policies share one streaming anomaly detector. The detector's standardization stays calibrated at every sampling cadence.
2. **Whether an anomaly is a sensor fault or a real event.** Feature-based classifiers are checked against expert-verified real events and real out-of-water fault records, not only synthetic data.
3. **Where a contamination event entered the river.** Station samples are inverted under a one-dimensional advection–dispersion model, with release time, mass, velocity and dispersion all unknown.

### Main results (from `results/summary.json`)

- **Detection:** escalating from 30- to 15-min sampling did not detect more injected events than fixed 30-min sampling (recall 0.399 vs 0.403; 1,500 paired trials; exact McNemar p = 0.55).
- **Localization:** escalation halved median source-localization error (54 m vs 100 m) for 3% more samples. Escalating 30 to 5 min matched fixed 15-min accuracy (38 m vs 36 m) with 43% fewer samples.
- **Real-data checks:** detectors and classifiers built to resist single-channel sensor faults missed or mislabelled the record's turbidity-led real events.

### Contents

| Path | Contents |
|---|---|
| `src/wqcadence/` | Package: data loading, detector, policies, injection, features, transport model, statistics, experiments |
| `tests/` | 80 automated tests (`python -m pytest`) |
| `scripts/` | `run_experiments.py`, `make_figures.py`, `build_paper.py`, `package_submission.py` |
| `results/` | Raw per-trial outputs (CSV) and `summary.json` |
| `figures/` | All manuscript figures (PDF and PNG) and the graphical abstract |
| `data/raw/` | Owenmore River source data from Cronin et al. (2025), unmodified (MIT licence) |

### Reproduce

```bash
python -m pip install -r requirements.txt
python -m pytest                       # ~10 s
python scripts/run_experiments.py      # ~10 min on 8 cores; fixed seeds
python scripts/make_figures.py
```

### Data and licence

Code: MIT licence. Source data: Cronin, L., Taylor, C.M., Briciu Burghina, C., Lucy, F.E., Regan, F. (2025), *MethodsX* 15:103538, redistributed unmodified under its MIT licence (see `data/raw/LICENSE_source_data.txt`).

### Note

The project was developed under the working name "AquaSentinel" and renamed WQCadence before release, to avoid confusion with an unrelated existing water-monitoring product of that name.
