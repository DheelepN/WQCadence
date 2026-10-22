"""Build every manuscript figure from results/ (and, for Figs 1-2, the data).

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wqcadence import pillar1 as p1  # noqa: E402
from wqcadence.data import event_runs, load_owenmore, model_matrix, quiet_mask  # noqa: E402
from wqcadence.injection import inject_event, robust_sd  # noqa: E402
from wqcadence.policies import run_policy  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"

# Reference palette (light mode), fixed role per entity across every figure.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
POLICY = {"Fixed-30": ORANGE, "Adaptive-30/15": BLUE, "Continuous-15": AQUA}
EVENT_SHADE = "#f0efec"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 1.5, "savefig.dpi": 300, "figure.dpi": 110,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType fonts (journal artwork rules)
})


def panel(ax, letter):
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig1_data(data):
    labels = [("temp", "Temperature (°C)"), ("spcond", "Specific conductance (µS/cm)"),
              ("ph", "pH"), ("turbidity", "Turbidity (NTU, log)")]
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.4), sharex=True)
    s = data.series
    for ax, (col, lab) in zip(axes, labels):
        for k in (1, 2):
            d = data.deployment(k)
            for a, b in event_runs(d.event_verified.to_numpy()):
                ax.axvspan(d.datetime[a], d.datetime[b], color=EVENT_SHADE, lw=0, zorder=0)
            ax.plot(d.datetime, d[col], color=BLUE, lw=0.9)
        if col == "turbidity":
            ax.set_yscale("log")
        ax.set_ylabel(lab, fontsize=7.5)
    for f in data.faults:
        raw = data.raw[f.deployment]
        r = raw[(raw.datetime >= f.start) & (raw.datetime <= f.end)]
        axes[1].scatter(r.datetime, r.spcond, marker="x", s=22, color=INK, lw=1.1, zorder=3)
    axes[1].annotate("out-of-water records\n(real faults, n = 11)", xy=(data.faults[1].start, 20),
                     xytext=(15, 60), textcoords="offset points", fontsize=7, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[0].text(0.01, 1.04, "Shaded: expert-verified events (Cronin et al., 2025)", transform=axes[0].transAxes,
                 fontsize=7.5, color=INK2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.align_ylabels(axes)
    save(fig, "fig1_owenmore_data")


def fig2_example(data):
    """One illustrative injection trial (deployment 2), all three policies."""
    df = data.deployment(2)
    X, q = model_matrix(df), quiet_mask(df)
    onsets = p1.candidate_onsets(q)
    onset = int(onsets[len(onsets) // 3])
    ev = {"severity": "moderate", "shape": "ramp_plateau", "amplitude": 2.2, "duration": 8}
    Xi = inject_event(X, onset, ev, robust_sd(X[onset - 96:onset]))
    lo, hi = onset - 24, onset + 24
    t = (np.arange(lo, hi) - onset) * 15 / 60
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
    ax0.axvspan(0, ev["duration"] * 0.25, color=EVENT_SHADE, lw=0)
    ax0.plot(t, np.expm1(Xi[lo:hi, 3]), color=BLUE, lw=1.4, label="with injected event")
    ax0.plot(t, np.expm1(X[lo:hi, 3]), color=MUTED, lw=1.0, ls="--", label="real background")
    ax0.set_ylabel("Turbidity (NTU)")
    ax0.legend(loc="upper left", ncols=2)
    ax0.set_title("Injected moderate event (2.2 local SD, 2 h, ramp-plateau)")
    ax1.axvspan(0, ev["duration"] * 0.25, color=EVENT_SHADE, lw=0)
    for i, p in enumerate(p1.MAIN):
        run = run_policy(Xi[onset - p1.WARMUP:hi], p, phase=0)
        idx = run.sample_idx + onset - p1.WARMUP
        m = (idx >= lo) & (idx < hi)
        ax1.plot((idx[m] - onset) * 0.25, run.scores[m], marker="os^"[i], ms=4.5, lw=1.0,
                 color=POLICY[p.name], label=p.name, mec="white", mew=0.6)
        al = np.flatnonzero(run.alarm[p1.WARMUP - 24:p1.WARMUP + 24])
        if al.size:
            ax1.plot([(al[0] - 24) * 0.25], [5.4 - 0.4 * i], marker="|", ms=9, mew=2, color=POLICY[p.name])
    for tau, name in ((p1.ADAPTIVE.tau1, "τ₁ alert"), (p1.ADAPTIVE.tau2, "τ₂ critical")):
        ax1.axhline(tau, color=INK2, lw=0.8, ls=":")
        ax1.text(t[0], tau + 0.08, name, fontsize=7, color=INK2)
    ax1.text(0.75, 5.0, "first alarm per policy", fontsize=7, color=INK2, va="center")
    ax1.set_ylim(0, 6.2)
    ax1.set_ylabel("Anomaly score (mean |z|)")
    ax1.set_xlabel("Hours from injected onset")
    ax1.legend(loc="upper right", ncols=1)
    save(fig, "fig2_policy_example")


def fig3_recall_cost(summary):
    r = summary["pillar1"]["recall"]
    sev = ["brief", "moderate", "severe"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
    w = 0.26
    for i, name in enumerate(POLICY):
        vals = [r[name][f"recall_{s}"] for s in sev]
        ci = np.array([r[name][f"ci_{s}"] for s in sev]).T
        x = np.arange(3) + (i - 1) * (w + 0.02)
        a.bar(x, vals, w, color=POLICY[name], label=name, edgecolor="white", linewidth=1)
        a.errorbar(x, vals, yerr=[np.array(vals) - ci[0], ci[1] - np.array(vals)], fmt="none", ecolor=INK2,
                   elinewidth=0.8, capsize=2)
    a.set_xticks(range(3), ["Brief/faint", "Moderate", "Severe"])
    a.set_ylabel("Detection recall")
    a.set_ylim(0, 1.05)
    a.legend(loc="upper left", fontsize=7.2)
    a.set_title("Recall by severity (95% Wilson CI)")
    panel(a, "a")

    bg = pd.DataFrame(summary["pillar1"]["background"])
    bg = bg[bg.persist == 2].set_index("policy")
    for fam, col, mk in (("Fixed", ORANGE, "o"), ("Adaptive", BLUE, "^")):
        names = [n for n in r if n.startswith(fam) and "[" not in n]
        names = sorted(names, key=lambda n: bg.loc[n, "cycles_per_day"])
        xs = [bg.loc[n, "cycles_per_day"] for n in names]
        ys = [r[n]["recall_moderate"] for n in names]
        b.plot(xs, ys, marker=mk, color=col, ms=5, mec="white", mew=0.6, label=f"{fam} family")
        for n, x_, y_ in zip(names, xs, ys):
            if fam == "Fixed":
                b.annotate(n.split("-")[1], (x_, y_), xytext=(-3, 6), textcoords="offset points",
                           fontsize=6.5, color=ORANGE, ha="right")
            else:
                b.annotate(n.split("-")[1], (x_, y_), xytext=(4, -10), textcoords="offset points",
                           fontsize=6.5, color=BLUE, ha="left")
    c = "Continuous-15"
    b.annotate("(= Continuous)", (bg.loc[c, "cycles_per_day"], r[c]["recall_moderate"]),
               xytext=(-6, -14), textcoords="offset points", fontsize=6.5, color=INK2, ha="right")
    b.set_xlabel("Measurement cycles per day")
    b.set_ylabel("Recall, moderate events")
    b.set_title("Detection–cost trade-off")
    b.legend(loc="lower right", fontsize=7.2)
    panel(b, "b")
    fig.tight_layout()
    save(fig, "fig3_recall_cost")


def fig4_tau(summary):
    tau = pd.DataFrame(summary["pillar1"]["tau_sweep"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for name, g in tau.groupby("policy", sort=False):
        g = g.sort_values("tau1")
        a.plot(g.tau1, g.recall_moderate, marker="o", ms=4, color=POLICY[name], label=name, mec="white", mew=0.5)
        b.plot(g.tau1, g.false_episodes_per_day, marker="o", ms=4, color=POLICY[name], label=name, mec="white", mew=0.5)
    for ax in (a, b):
        ax.axvline(1.8, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("Alert threshold τ₁")
    a.text(1.82, 0.02, "a priori", fontsize=7, color=INK2)
    a.set_ylabel("Recall, moderate events")
    a.set_ylim(0, 1.02)
    a.set_title("Detection vs threshold")
    b.set_ylabel("False-alarm episodes per quiet day")
    b.set_title("Real-background false alarms vs threshold")
    a.legend(fontsize=7.2)
    panel(a, "a"); panel(b, "b")
    fig.tight_layout()
    save(fig, "fig4_threshold_sweep")


def fig5_classifier(summary):
    des = summary["pillar2"]["designs"]
    keys = ["discharge_only|F4", "discharge_only|F6", "discharge_and_runoff|F4", "discharge_and_runoff|F6"]
    labels = ["F4, discharge\n(a priori)", "F6, discharge", "F4, discharge\n+ runoff", "F6, discharge\n+ runoff"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1, 1.2]})
    x = np.arange(4)
    for i, (m, col, lab) in enumerate((("logistic_cv", BLUE, "Logistic"), ("forest_cv", ORANGE, "Random forest"))):
        v = [des[k][m]["f1_mean"] for k in keys]
        e = [des[k][m]["f1_sd"] for k in keys]
        a.bar(x + (i - 0.5) * 0.36, v, 0.34, color=col, label=lab, edgecolor="white", linewidth=1)
        a.errorbar(x + (i - 0.5) * 0.36, v, yerr=e, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    rule = [des[k]["rule"]["f1"] for k in keys]
    a.scatter(x, rule, marker="_", s=180, color=INK, lw=1.6, zorder=3, label="Rule: fault if ≤1 channel deviates")
    a.set_xticks(x, labels, fontsize=6.8)
    a.set_ylabel("Fault-class F1 (5×10 CV, mean ± SD)")
    a.set_ylim(0, 1.22)
    a.set_yticks(np.arange(0, 1.01, 0.2))
    a.legend(loc="upper left", fontsize=6.6, ncols=3, handlelength=1.2, columnspacing=0.8)
    a.set_title("Synthetic cross-validation")
    panel(a, "a")

    ext = pd.DataFrame(des[keys[0]]["external"])
    names = ext.window.tolist()
    shown = ((keys[0], BLUE, "o", "F4, discharge-only training (a priori)"),
             (keys[3], ORANGE, "^", "F6, discharge + runoff training (revised)"))
    for j, (k, col, mk, lab) in enumerate(shown):
        e = pd.DataFrame(des[k]["external"]).set_index("window").loc[names]
        b.scatter(e.p_fault, np.arange(len(names)) + (j - 0.5) * 0.22, color=col, marker=mk, s=26,
                  edgecolor="white", linewidth=0.5, label=lab, zorder=3)
    b.axvline(0.5, color=INK2, lw=0.8, ls=":")
    kinds = ext.kind.tolist()
    b.set_yticks(range(len(names)), [f"{'Fault' if k == 'real_fault' else 'Event'} · {n}" for k, n in zip(kinds, names)],
                 fontsize=6.8)
    nf = sum(k == "real_fault" for k in kinds)
    b.axhspan(-0.5, nf - 0.5, color=EVENT_SHADE, lw=0, zorder=0)
    b.set_xlim(0, 1)
    b.invert_yaxis()
    b.set_xlabel("Predicted probability of fault")
    b.set_title("Real windows (shaded: true faults)")
    b.legend(loc="upper center", bbox_to_anchor=(0.3, -0.2), ncols=1, fontsize=6.8)
    panel(b, "b")
    fig.tight_layout()
    save(fig, "fig5_fault_classifier")


def fig6_localization(summary):
    by = summary["pillar3"]["by_cadence"]
    sc = pd.read_csv(RES / "pillar3_scenarios.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.15, 1, 0.9]})
    for fam, col, mk in (("Fixed", ORANGE, "o"), ("Adaptive", BLUE, "^")):
        names = sorted([n for n in by if n.startswith(fam)], key=lambda n: by[n]["samples_per_station_day"])
        xs = [by[n]["samples_per_station_day"] for n in names]
        ys = [by[n]["inversion_median"] for n in names]
        ci = np.array([by[n]["inversion_median_ci"] for n in names]).T
        a.errorbar(xs, ys, yerr=[np.array(ys) - ci[0], ci[1] - np.array(ys)], fmt=mk + ("-" if fam == "Fixed" else ""),
                   color=col, ms=5, mec="white", mew=0.6, ecolor=col, elinewidth=0.8, capsize=2, label=fam)
        for n, x_, y_ in zip(names, xs, ys):
            a.annotate(n.split("-")[1], (x_, y_), xytext=(5, 3), textcoords="offset points", fontsize=6.5, color=INK2)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("Samples per station per day")
    a.set_ylabel("Median source error (m)")
    a.set_title("Error vs sampling cost")
    a.legend(fontsize=7)
    panel(a, "a")

    for n, col, ls in (("Fixed-30", ORANGE, "-"), ("Adaptive-30/15", BLUE, "-"), ("Adaptive-30/5", BLUE, "--"),
                       ("Fixed-15", AQUA, "-")):
        e = np.sort(sc.loc[sc.cadence == n, "inversion_error"].dropna().to_numpy())
        b.plot(e, np.arange(1, e.size + 1) / len(sc[sc.cadence == n]), color=col, ls=ls, label=n, lw=1.4)
    b.set_xscale("log")
    b.set_xlim(1, 1e4)
    b.set_xlabel("Source error (m)")
    b.set_ylabel("Cumulative fraction of scenarios")
    b.set_title("Error distribution")
    b.legend(fontsize=6.6, loc="upper left")
    panel(b, "b")

    st = pd.DataFrame(summary["pillar3"]["station_count"])
    for n, col, mk in (("Fixed-30", ORANGE, "o"), ("Adaptive-30/15", BLUE, "^")):
        g = st[st.cadence == n].sort_values("n_stations")
        c.plot(g.n_stations, g.median_error, marker=mk, color=col, ms=5, mec="white", mew=0.6, label=n)
    c.set_xlabel("Stations on reach")
    c.set_ylabel("Median source error (m)")
    c.set_title("Station count")
    c.legend(fontsize=6.6)
    panel(c, "c")
    fig.tight_layout()
    save(fig, "fig6_localization")


def graphical_abstract(summary):
    """Required by the journal: >= 1328 px wide, readable at 13 cm. Drawn directly from the results."""
    r, by = summary["pillar1"]["recall"], summary["pillar3"]["by_cadence"]
    fig = plt.figure(figsize=(5.2, 2.95))
    fig.text(0.02, 0.95, "Adaptive sampling pays off after detection, not for it", fontsize=10.5,
             fontweight="bold", va="top")
    a = fig.add_axes([0.08, 0.25, 0.33, 0.52])
    b = fig.add_axes([0.53, 0.25, 0.45, 0.52])
    nl = chr(10)
    det = [(f"Fixed{nl}30 min", r["Fixed-30"]["recall"], ORANGE),
           (f"Adaptive{nl}30→15", r["Adaptive-30/15"]["recall"], BLUE),
           (f"Fixed{nl}15 min", r["Continuous-15"]["recall"], AQUA)]
    for i, (lab, v, col) in enumerate(det):
        a.bar(i, v, 0.66, color=col, edgecolor="white", linewidth=1)
        a.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    a.set_xticks(range(3), [d[0] for d in det], fontsize=7)
    a.set_ylim(0, 0.7)
    a.set_yticks([0, 0.2, 0.4, 0.6])
    a.tick_params(labelsize=7)
    a.set_title(f"Detection recall{nl}(1,500 paired trials)", fontsize=8)
    loc = [(f"Fixed{nl}30 min", "Fixed-30", ORANGE, None), (f"Adaptive{nl}30→15", "Adaptive-30/15", BLUE, None),
           (f"Fixed{nl}15 min", "Fixed-15", AQUA, None), (f"Adaptive{nl}30→5", "Adaptive-30/5", BLUE, "////")]
    for i, (lab, k, col, hatch) in enumerate(loc):
        v = by[k]["inversion_median"]
        b.bar(i, v, 0.66, color=col, edgecolor="white", linewidth=1, hatch=hatch)
        b.text(i, v + 2.5, f"{v:.0f} m", ha="center", fontsize=7.5, color=INK)
    labels = [f"{l[0]}{nl}{by[l[1]]['samples_per_station_day']:.0f}/day" for l in loc]
    b.set_xticks(range(4), labels, fontsize=7)
    b.tick_params(labelsize=7)
    b.set_ylim(0, 120)
    b.set_title(f"Source-localization error (median),{nl}release time unknown",
                fontsize=8)
    fig.text(0.02, 0.015, "Owenmore River data · Monte Carlo ground truth · advection–dispersion simulation · "
             "/day = samples per station-day", fontsize=6.2, color=INK2)
    for ext in ("pdf", "png", "tiff"):
        fig.savefig(FIG / f"graphical_abstract.{ext}", dpi=400, facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    summary = json.loads((RES / "summary.json").read_text())
    data = load_owenmore(ROOT / "data" / "raw")
    fig1_data(data)
    fig2_example(data)
    fig3_recall_cost(summary)
    fig4_tau(summary)
    fig5_classifier(summary)
    fig6_localization(summary)
    graphical_abstract(summary)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
