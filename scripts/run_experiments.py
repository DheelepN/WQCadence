"""Run every experiment in the paper and write raw outputs + summary statistics.

    python scripts/run_experiments.py            # full run (~10 min on 8 cores)
    python scripts/run_experiments.py --quick    # smoke run

Outputs go to results/. Every number quoted in the manuscript is read from
results/summary.json by scripts/build_paper.py; nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wqcadence import __version__  # noqa: E402
from wqcadence import pillar1 as p1  # noqa: E402
from wqcadence import pillar2 as p2  # noqa: E402
from wqcadence import pillar3 as p3  # noqa: E402
from wqcadence.data import event_runs, load_owenmore, quiet_mask  # noqa: E402
from wqcadence.injection import SEVERITY  # noqa: E402
from wqcadence.stats import bootstrap_ci, holm, mcnemar_exact, wilcoxon_paired, wilson  # noqa: E402

SEEDS = {"pillar1": 20260811, "pillar1_tau": 20260814, "pillar2": 20260812, "pillar3": 20260813,
         "pillar3_stations": 20260815}
TAU1_GRID = (1.4, 1.6, 1.8, 2.0, 2.2, 2.6)
OUT = ROOT / "results"


def clean_keys(o):
    """JSON needs string keys; pandas summaries produce tuples and numpy ints."""
    if isinstance(o, dict):
        return {(" | ".join(map(str, k)) if isinstance(k, tuple) else str(k)): clean_keys(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_keys(v) for v in o]
    return o


def unique(policies):
    seen, out = set(), []
    for p in policies:
        if p.name not in seen:
            seen.add(p.name)
            out.append(p)
    return tuple(out)


def recall_table(df: pd.DataFrame, names) -> dict:
    out = {}
    for n in names:
        d = df[f"{n}|detected"]
        row = {"recall": float(d.mean()), "ci": wilson(int(d.sum()), len(d)),
               "latency_median": float(df[f"{n}|latency"].median()),
               "latency_p90": float(df[f"{n}|latency"].quantile(0.9)),
               "cycles_mean": float(df[f"{n}|cycles"].mean()),
               "critical_rate": float(df[f"{n}|critical"].mean())}
        for sev in SEVERITY:
            s = df.loc[df.severity == sev, f"{n}|detected"]
            row[f"recall_{sev}"] = float(s.mean())
            row[f"ci_{sev}"] = wilson(int(s.sum()), len(s))
            row[f"n_{sev}"] = int(len(s))
        out[n] = row
    return out


def paired_policy_tests(df: pd.DataFrame, a: str, b: str) -> dict:
    res = {"recall": mcnemar_exact(df[f"{a}|detected"], df[f"{b}|detected"])}
    for sev in SEVERITY:
        s = df[df.severity == sev]
        res[f"recall_{sev}"] = mcnemar_exact(s[f"{a}|detected"], s[f"{b}|detected"])
    both = df[df[f"{a}|detected"] & df[f"{b}|detected"]]
    lat = wilcoxon_paired(both[f"{a}|latency"], both[f"{b}|latency"])
    lat["ci_mean_diff"] = bootstrap_ci((both[f"{a}|latency"] - both[f"{b}|latency"]).to_numpy(), seed=1)
    res["latency"] = lat
    return res


def detector_calibration(seed: int = 7) -> dict:
    """E[z^2] and P(|z|>3) on stationary Gaussian noise, by sampling gap, for
    the detector used here and for the naive incremental EW variance."""
    from wqcadence.detector import GapAwareEWMA
    out = {}
    for gap in (1, 2, 4, 8):
        x = np.random.default_rng(seed + gap).normal(0, 1, (40_000, 4))
        det = GapAwareEWMA(sd_floor=np.full(4, 1e-6))
        z = np.array([det.update(r, gap) for r in x[::gap]])[200:]
        r, m, v, zo = 0.5 ** (1 / 6), x[0].copy(), np.zeros(4), []
        for row in x[::gap][1:]:
            zo.append((row - m) / np.sqrt(np.maximum(v, 1e-12)))
            ret, d = r ** gap, row - m
            m, v = m + (1 - ret) * d, ret * (v + (1 - ret) * d * d)
        zo = np.clip(np.array(zo[200:]), -10, 10)
        out[str(gap)] = {"z2": float(np.mean(z ** 2)), "p_gt3": float(np.mean(np.abs(z) > 3)),
                         "naive_z2": float(np.mean(zo ** 2)), "naive_p_gt3": float(np.mean(np.abs(zo) > 3))}
    return out


def pillar1(data, n_trials: int, n_tau: int) -> dict:
    t = time.time()
    rms = p1.with_aggregate(p1.MAIN, "rms")
    policies = unique(p1.MAIN + p1.SWEEP + rms)
    trials = p1.injection_trials(data, n_trials, SEEDS["pillar1"], policies)
    trials.to_csv(OUT / "pillar1_injection_trials.csv", index=False)

    names = [p.name for p in policies]
    rec = recall_table(trials, names)
    tests = {
        "Adaptive_vs_Fixed30": paired_policy_tests(trials, p1.ADAPTIVE.name, p1.FIXED_30.name),
        "Adaptive_vs_Continuous": paired_policy_tests(trials, p1.ADAPTIVE.name, p1.CONTINUOUS.name),
        "Continuous_vs_Fixed30": paired_policy_tests(trials, p1.CONTINUOUS.name, p1.FIXED_30.name),
    }
    holm_in = {f"{k}|{m}": v[m]["p"] for k, v in tests.items() for m in ("recall", "latency")}
    holm_adj = holm(holm_in)

    bg = pd.concat([p1.background(data, policies),
                    p1.background(data, p1.with_persist(p1.MAIN + rms, 1))], ignore_index=True)
    bg.to_csv(OUT / "pillar1_background_runs.csv", index=False)
    bgs = p1.summarise_background(bg)
    bgs.to_csv(OUT / "pillar1_background_summary.csv", index=False)

    real = p1.real_event_detection(data, p1.MAIN + rms)
    real.to_csv(OUT / "pillar1_real_event_detection.csv", index=False)
    real_sum = real.groupby(["policy", "event"]).detected.mean().unstack().round(3)

    # Threshold sweep: operating characteristics of the three main policies.
    tau_rows = []
    for tau in TAU1_GRID:
        pol = p1.with_tau1(p1.MAIN, tau)
        tr = p1.injection_trials(data, n_tau, SEEDS["pillar1_tau"], pol)
        bgt = p1.summarise_background(p1.background(data, pol))
        for p in pol:
            b = bgt[bgt.policy == p.name].iloc[0]
            tau_rows.append({"policy": p.name.split("@")[0], "tau1": tau,
                             "recall": float(tr[f"{p.name}|detected"].mean()),
                             "recall_moderate": float(tr.loc[tr.severity == "moderate", f"{p.name}|detected"].mean()),
                             "latency_median": float(tr[f"{p.name}|latency"].median()),
                             "far_per_1000": float(b.far_per_1000),
                             "false_episodes_per_day": float(b.false_episodes_per_day),
                             "cycles_per_day": float(b.cycles_per_day)})
    tau_df = pd.DataFrame(tau_rows)
    tau_df.to_csv(OUT / "pillar1_tau_sweep.csv", index=False)

    q = {k: int(quiet_mask(data.deployment(k)).sum()) for k in (1, 2)}
    return {
        "n_trials": n_trials, "n_tau_trials": n_tau, "seconds": round(time.time() - t, 1),
        "candidate_onsets": {k: int(len(p1.candidate_onsets(quiet_mask(data.deployment(k))))) for k in (1, 2)},
        "trials_by_deployment": trials.deployment.value_counts().sort_index().to_dict(),
        "quiet_steps": q,
        "recall": rec, "tests": tests, "holm": holm_adj,
        "background": bgs.to_dict(orient="records"),
        "real_events": {"by_policy": real_sum.to_dict(orient="index"),
                        "events": sorted(real.event.unique().tolist())},
        "tau_sweep": tau_rows,
    }


def pillar2(data, n_per_class: int, n_repeats: int) -> dict:
    t = time.time()
    real = p2.real_windows(data)
    seed = SEEDS["pillar2"]
    out = {"n_per_class": n_per_class, "cv": "5-fold stratified, repeated", "n_repeats": n_repeats,
           "designs": {}, "real_windows": [{k: r[k] for k in ("name", "kind", "n")} for r in real]}
    ext_frames = []
    for design, sigs in (("discharge_only", ("discharge",)), ("discharge_and_runoff", ("discharge", "runoff"))):
        pool = p2.build_pool(data, n_per_class, seed, event_signatures=sigs)
        y = np.array([p["label"] for p in pool])
        fouling = p2.build_pool(data, 100, seed + 1, event_signatures=sigs, fault_types=("fouling",))
        fouling = [p for p in fouling if p["label"] == 1]
        for fset in ("F4", "F6"):
            X = p2.featurise(pool, fset)
            key = f"{design}|{fset}"
            lr = p2.logistic().fit(X, y)
            res = {
                "logistic_cv": p2.cross_validate(X, y, p2.logistic, seed, n_repeats=n_repeats),
                "forest_cv": p2.cross_validate(X, y, lambda: p2.random_forest(seed), seed, n_repeats=n_repeats),
                "rule": p2.rule_baseline(X, y),
                "leave_one_type_out_recall": p2.leave_one_type_out(pool, X, p2.logistic),
                "unseen_fouling_recall": float(np.mean(lr.predict(p2.featurise(fouling, fset)) == 1)),
                "coefficients": p2.coefficients(pool, fset),
            }
            ext = p2.external_check(pool, real, fset, p2.logistic)
            ext.insert(0, "design", key)
            ext_frames.append(ext)
            res["external"] = ext[["window", "kind", "records", "p_fault", "predicted", "correct"]].to_dict(orient="records")
            res["external_correct"] = {"real_fault": int(ext[ext.kind == "real_fault"].correct.sum()),
                                       "real_event": int(ext[ext.kind == "real_event"].correct.sum())}
            out["designs"][key] = res
    pd.concat(ext_frames).to_csv(OUT / "pillar2_external_windows.csv", index=False)
    out["seconds"] = round(time.time() - t, 1)
    return out


def pillar3(n_scen: int, n_station_scen: int) -> dict:
    t = time.time()
    df = p3.run(n_scen, SEEDS["pillar3"])
    df.to_csv(OUT / "pillar3_scenarios.csv", index=False)
    summary = {}
    for cad, g in df.groupby("cadence", sort=False):
        e = g.inversion_error.dropna()
        o = g.oracle_error.dropna()
        summary[cad] = {
            "localized_rate": float(g.inversion_error.notna().mean()),
            "inversion_median": float(e.median()), "inversion_median_ci": bootstrap_ci(e.to_numpy(), np.median, seed=2),
            "inversion_p90": float(e.quantile(0.9)),
            "inversion_within_100m": float((e <= 100).mean()),
            "oracle_median": float(o.median()), "oracle_p90": float(o.quantile(0.9)),
            "samples_per_station_day": float(g.samples_per_station_day.mean()),
        }
    wide = df.pivot(index="scenario", columns="cadence", values="inversion_error")
    comparisons = {}
    for a, b in (("Adaptive-30/15", "Fixed-30"), ("Adaptive-30/5", "Fixed-30"), ("Adaptive-30/5", "Fixed-15"),
                 ("Fixed-15", "Fixed-30"), ("Fixed-5", "Fixed-15")):
        both = wide[[a, b]].dropna()
        w = wilcoxon_paired(both[a], both[b])
        w["ci_median_diff"] = bootstrap_ci((both[a] - both[b]).to_numpy(), np.median, seed=3)
        w["a_better_fraction"] = float((both[a] < both[b]).mean())
        comparisons[f"{a} vs {b}"] = w
    adj = holm({k: v["p"] for k, v in comparisons.items()})
    for k in comparisons:
        comparisons[k]["p_holm"] = adj[k]

    st = p3.run_station_count(n_station_scen, SEEDS["pillar3_stations"])
    st.to_csv(OUT / "pillar3_station_count.csv", index=False)
    st_sum = (st.groupby(["cadence", "n_stations"])
                .agg(localized_rate=("inversion_error", lambda s: float(s.notna().mean())),
                     median_error=("inversion_error", "median"),
                     n_detecting_mean=("n_detecting", "mean"))
                .reset_index().to_dict(orient="records"))
    return {"n_scenarios": n_scen, "n_station_scenarios": n_station_scen, "by_cadence": summary,
            "comparisons": comparisons, "station_count": st_sum, "seconds": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    n1, ntau, n2, rep, n3, n3s = (60, 30, 40, 2, 12, 6) if args.quick else (1500, 600, 250, 10, 500, 200)

    data = load_owenmore(ROOT / "data" / "raw")
    flagged = data.series.event_verified.to_numpy()
    summary = {
        "meta": {"package_version": __version__, "python": platform.python_version(), "seeds": SEEDS,
                 "quick": args.quick, "numpy": np.__version__, "pandas": pd.__version__},
        "data": {
            "records": {int(k): int(v) for k, v in data.series.groupby("deployment").size().items()},
            "span": {int(k): [str(g.datetime.min()), str(g.datetime.max())] for k, g in data.series.groupby("deployment")},
            "sct_flagged": int(data.series.event_sct.sum()),
            "verified_flagged": int(flagged.sum()),
            "verified_events": [[str(data.series.datetime[s]), int(e - s + 1)] for s, e in event_runs(flagged)],
            "faults": [{"deployment": f.deployment, "start": str(f.start), "end": str(f.end), "records": len(f.window),
                        "baseline_side": f.baseline_side} for f in data.faults],
            "channel_stats": data.series.groupby("deployment")[["temp", "spcond", "ph", "turbidity"]]
                                 .describe().round(2).to_dict(),
        },
    }
    summary["detector_calibration"] = detector_calibration()
    print("pillar 1 ...", flush=True)
    summary["pillar1"] = pillar1(data, n1, ntau)
    print("pillar 2 ...", flush=True)
    summary["pillar2"] = pillar2(data, n2, rep)
    print("pillar 3 ...", flush=True)
    summary["pillar3"] = pillar3(n3, n3s)
    name = "summary_quick.json" if args.quick else "summary.json"
    (OUT / name).write_text(json.dumps(clean_keys(summary), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("wrote", OUT / name)


if __name__ == "__main__":
    main()
