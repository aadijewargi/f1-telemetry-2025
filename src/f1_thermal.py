#!/usr/bin/env python3
"""
F1 2025: brake-disc and tyre thermal models from qualifying traces.

Usage:  python f1_thermal.py <data_folder> <out_folder>    (run f1_analysis.py first)

A. Brake disc (lumped heat balance for one front disc, lap-periodic steady state)
     m c(T) dT/dt = P_friction(t) - h(v) A (T - T_air) - eps sigma A (T^4 - T_air^4)
   P_friction = car deceleration power - aerodynamic drag power - MGU-K recovery, split front/rear.
   Disc mass, area, drag area and the cooling-scale are ASSUMED (no brake-temperature channel is public).
   The cooling scale is calibrated so the median circuit's mean disc temperature is 600 C, so absolute
   temperatures are not predictions; the between-circuit RANKING is the result, and is checked against
   parameter sensitivity.
B. Tyre: a physics-motivated load index (dissipated work ~ F^2/F_max * v) per circuit, tested against the
   observed tyre degradation of each circuit.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import signal, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
import f1_analysis as fa

OUT, TAB, FIG = fa.OUT, fa.OUT / "tables", fa.OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, MUTED, GRID, AXIS = fa.SURF, fa.INK, fa.INK2, fa.MUTED, fa.GRID, fa.AXIS
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
S = {}
SIGMA = 5.670374e-8

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
                     "axes.axisbelow": True})


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=170)
    plt.close(fig)


# assumed parameters (documented in the report)
BASE = dict(cda=1.1, disc_kg=1.4, front_share=0.60, cool=1.0, area=0.12, emissivity=0.8)
M_CAR = fa.M_QUALI


def pole_lap(g):
    drv = g.groupby("Driver").LapTime_s.first().idxmin()
    return drv, g[g.Driver == drv].sort_values("Distance_m")


def brake_power_series(lap, rho, cda, dt=0.05):
    """Uniform time grid: speed, friction-brake power (total car, W), brake flag."""
    t = lap.Time_s.to_numpy()
    tt = np.arange(0, t[-1], dt)
    v = np.interp(tt, t, lap.Speed.to_numpy() / 3.6)
    brk = np.interp(tt, t, lap.Brake.to_numpy()) > 0.5
    vs = signal.savgol_filter(v, 9, 2)
    a = np.gradient(vs, dt)
    p_dec = -M_CAR * a * vs                                  # W released by deceleration (positive when slowing)
    p_drag = 0.5 * rho * cda * vs ** 3
    p = np.where(brk, np.maximum(p_dec - p_drag, 0), 0.0)
    # MGU-K: 120 kW until 2 MJ per lap have been recovered
    left, mg = 2.0e6, np.zeros_like(p)
    for i in range(len(p)):
        if p[i] > 0 and left > 0:
            mg[i] = min(120e3, p[i], left / dt)
            left -= mg[i] * dt
    return tt, vs, p - mg, brk


def disc_temperature(tt, v, pf, t_air, par, laps=8):
    dt = tt[1] - tt[0]
    n = len(tt)
    P = par["front_share"] / 2 * pf                          # W into one front disc
    T = t_air + 273.15
    Tair = t_air + 273.15
    out = None
    for lap in range(laps):
        trace = np.empty(n)
        for i in range(n):
            c = 800.0 + 0.9 * (T - 273.15)                   # J/kgK, carbon-carbon, rises with temperature
            h = (25.0 + 6.5 * max(v[i], 5.0) ** 0.8) * par["cool"]
            q = h * par["area"] * (T - Tair) + par["emissivity"] * SIGMA * par["area"] * (T ** 4 - Tair ** 4)
            T += dt * (P[i] - q) / (par["disc_kg"] * c)
            trace[i] = T - 273.15
        out = trace
    return out


def calibrate_cooling(cases, target=600.0):
    lo, hi = 0.3, 6.0
    for _ in range(18):
        mid = (lo + hi) / 2
        means = [disc_temperature(tt, v, pf, ta, dict(BASE, cool=mid), laps=5).mean() for _, tt, v, pf, ta in cases]
        if np.median(means) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def brake_analysis(tr, wx, wet):
    wxi = wx.set_index("Round")
    cases = []
    for rnd, g in tr[~tr.Round.isin(wet)].groupby("Round"):
        drv, lap = pole_lap(g)
        if len(lap) < 100:
            continue
        rho, ta = float(wxi.loc[rnd, "AirDensity_kgm3"]), float(wxi.loc[rnd, "AirTemp_C"])
        tt, v, pf, brk = brake_power_series(lap, rho, BASE["cda"])
        cases.append((rnd, tt, v, pf, ta, drv, g.Event.iloc[0], rho))
    base_cases = [(c[0], c[1], c[2], c[3], c[4]) for c in cases]
    cool = calibrate_cooling(base_cases)
    S["brake_cooling_scale"] = float(cool)
    variants = {"baseline": dict(BASE, cool=cool),
                "drag area -30%": dict(BASE, cool=cool, cda=0.77), "drag area +30%": dict(BASE, cool=cool, cda=1.43),
                "disc mass -30%": dict(BASE, cool=cool, disc_kg=0.98), "disc mass +30%": dict(BASE, cool=cool, disc_kg=1.82),
                "cooling -30%": dict(BASE, cool=cool * .7), "cooling +30%": dict(BASE, cool=cool * 1.3),
                "front share 50%": dict(BASE, cool=cool, front_share=0.5), "front share 70%": dict(BASE, cool=cool, front_share=0.7)}
    rows, traces, sens = [], {}, {k: [] for k in variants}
    for rnd, tt, v, pf0, ta, drv, event, rho in cases:
        for name, par in variants.items():
            pf = pf0 if par["cda"] == BASE["cda"] else brake_power_series(
                tr[(tr.Round == rnd) & (tr.Driver == drv)].sort_values("Distance_m"), rho, par["cda"])[2]
            T = disc_temperature(tt, v, pf, ta, par, laps=6)
            sens[name].append((rnd, T.max(), T.mean()))
            if name == "baseline":
                traces[rnd] = (tt, T, v)
                rows.append(dict(Round=rnd, Event=event, driver=drv, lap_s=float(tt[-1]),
                                 friction_MJ_total=float(pf.sum() * (tt[1] - tt[0]) / 1e6),
                                 front_disc_MJ=float(par["front_share"] / 2 * pf.sum() * (tt[1] - tt[0]) / 1e6),
                                 peak_C=float(T.max()), mean_C=float(T.mean()), min_C=float(T.min()),
                                 swing_C=float(T.max() - T.min()), frac_above_800=float((T > 800).mean()),
                                 brake_frac=float((pf > 0).mean())))
    B = pd.DataFrame(rows)
    sp = {}
    base = pd.DataFrame(sens["baseline"], columns=["Round", "peak", "mean"]).set_index("Round")
    rngs = []
    for name, vals in sens.items():
        d = pd.DataFrame(vals, columns=["Round", "peak", "mean"]).set_index("Round")
        sp[name] = float(stats.spearmanr(base.peak, d.loc[base.index].peak).statistic)
        rngs.append(d.peak.rename(name))
    R = pd.concat(rngs, axis=1)
    B["peak_min_C"] = B.Round.map(R.min(axis=1))
    B["peak_max_C"] = B.Round.map(R.max(axis=1))
    B = B.sort_values("peak_C", ascending=False)
    B.to_csv(TAB / "thermal_brake_by_circuit.csv", index=False)
    pd.Series(sp).to_csv(TAB / "thermal_brake_rank_sensitivity.csv", header=["spearman_vs_baseline"])
    ce = pd.read_csv(TAB / "circuit_energy.csv").set_index("Round")
    B2 = B.set_index("Round")
    S["brake"] = dict(
        circuits=int(len(B)), cooling_scale=float(cool),
        peak_median_C=float(B.peak_C.median()), peak_min_C=float(B.peak_C.min()), peak_max_C=float(B.peak_C.max()),
        hottest=B.iloc[0].Event, coolest=B.iloc[-1].Event,
        swing_median_C=float(B.swing_C.median()), frac_above_800_max=float(B.frac_above_800.max()),
        friction_MJ_median=float(B.friction_MJ_total.median()),
        friction_share_of_braking_energy=float((B2.friction_MJ_total / ce.loc[B2.index].Ebrake_MJ).median()),
        rank_spearman_min=float(min(sp.values())), rank_spearman_median=float(np.median(list(sp.values()))),
        r_peak_vs_brakeenergy=float(stats.pearsonr(B2.friction_MJ_total, B2.peak_C).statistic),
        r_peak_vs_airtemp=float(stats.pearsonr(B2.peak_C, wx.set_index("Round").loc[B2.index].AirTemp_C).statistic))
    return B, traces, sp


# ------------------------------------------------------------------ tyre
def tyre_analysis(tr, wx, wet):
    K = fa.trace_kinematics(tr[~tr.Round.isin(wet)])
    K["a2"] = K.alat_g ** 2 + K.along_g ** 2
    K["v"] = K.v_kph / 3.6
    # load indices per lap, with 20 m sample spacing: sum(a^2 ds) [g^2 m] and the heating-rate proxy mean(a^2 v)
    K["ds"] = 20.0
    idx = K.groupby("Round").apply(lambda d: pd.Series(dict(
        load_g2m_per_lap=(d.a2 * d.ds).sum() / d.Driver.nunique(),
        heat_rate=(d.a2 * d.v).mean(),
        mean_a2=d.a2.mean(), high_g_share=(np.sqrt(d.a2) > 3).mean())), include_groups=False)
    wxi = wx.set_index("Round")
    idx["track_c"] = wxi.loc[idx.index].TrackTemp_C
    pm = pd.read_csv(TAB / "pace_model_by_round.csv").set_index("Round")
    D = idx.join(pm[["tyre_MEDIUM", "tyre_HARD", "tyre_SOFT"]])
    D.to_csv(TAB / "thermal_tyre_load_vs_degradation.csv")
    rows = []
    for comp in ("tyre_MEDIUM", "tyre_HARD", "tyre_SOFT"):
        for f in ("load_g2m_per_lap", "heat_rate", "mean_a2", "high_g_share", "track_c"):
            d = D[[f, comp]].dropna()
            if len(d) < 8:
                continue
            r = stats.pearsonr(d[f], d[comp])
            rs = stats.spearmanr(d[f], d[comp])
            rows.append(dict(compound=comp.split("_")[1], feature=f, n=len(d), r=r.statistic, p=r.pvalue,
                             rho=rs.statistic, p_spearman=rs.pvalue))
    T = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    T["p_holm"] = np.minimum(1, np.maximum.accumulate((len(T) - np.arange(len(T))) * T.p.to_numpy()))
    T.to_csv(TAB / "thermal_tyre_correlations.csv", index=False)
    best = T.iloc[0]
    S["tyre"] = dict(circuits=int(len(D)), tests=int(len(T)), best_feature=best.feature, best_compound=best.compound,
                     best_r=float(best.r), best_p=float(best.p), best_p_holm=float(best.p_holm),
                     sig_after_holm=int((T.p_holm < .05).sum()),
                     r_load_medium=float(T[(T.compound == "MEDIUM") & (T.feature == "load_g2m_per_lap")].r.iloc[0]),
                     r_track_medium=float(T[(T.compound == "MEDIUM") & (T.feature == "track_c")].r.iloc[0]),
                     r_load_hard=float(T[(T.compound == "HARD") & (T.feature == "load_g2m_per_lap")].r.iloc[0]))
    return D, T


# ------------------------------------------------------------------ figures
def fig_trace(B, traces):
    rows = [B.iloc[0], B.iloc[len(B) // 2], B.iloc[-1]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for r, col in zip(rows, (ORANGE, BLUE, AQUA)):
        tt, T, v = traces[int(r.Round)]
        ax.plot(tt / tt[-1] * 100, T, color=col, lw=2, label=r.Event.replace(" Grand Prix", ""))
    ax.set_xlabel("Lap progress (%)")
    ax.set_ylabel("Front brake disc temperature (C, modelled)")
    ax.set_title("Modelled disc temperature: hottest, median and coolest circuit")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    savefig(fig, "fig16_brake_temperature.png")


def fig_rank(B):
    b = B.sort_values("peak_C")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(b))
    ax.hlines(y, b.peak_min_C, b.peak_max_C, color=GRID, lw=5, zorder=1)
    ax.scatter(b.peak_C, y, s=60, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(b.Event.str.replace(" Grand Prix", ""))
    ax.set_xlabel("Peak front disc temperature (C): baseline, with range across parameter variations")
    ax.set_title("Which circuits are hardest on brakes")
    ax.grid(axis="y", visible=False)
    savefig(fig, "fig17_brake_ranking.png")


def fig_tyre(D):
    d = D[["load_g2m_per_lap", "tyre_MEDIUM"]].dropna()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(d.load_g2m_per_lap / 1e3, d.tyre_MEDIUM, s=64, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
    k = np.polyfit(d.load_g2m_per_lap / 1e3, d.tyre_MEDIUM, 1)
    xs = np.linspace((d.load_g2m_per_lap / 1e3).min(), (d.load_g2m_per_lap / 1e3).max(), 10)
    ax.plot(xs, np.polyval(k, xs), color=MUTED, lw=2)
    r = stats.pearsonr(d.load_g2m_per_lap, d.tyre_MEDIUM)
    ax.text(.97, .05, f"r = {r.statistic:.2f}, p = {r.pvalue:.2f}, n = {len(d)}", transform=ax.transAxes, color=INK2, ha="right")
    ax.set_xlabel("Tyre load index per lap (1000 g² m)")
    ax.set_ylabel("Observed medium-tyre degradation (s per lap)")
    ax.set_title("Does the physics load index predict tyre wear?")
    savefig(fig, "fig18_tyre_load_vs_degradation.png")


if __name__ == "__main__":
    d = fa.load()
    wx = d["weather"]
    wet = set(wx.loc[wx.RainFraction >= fa.WET_RAIN_FRACTION, "Round"])
    fa.R["wet_rounds"] = sorted(int(x) for x in wet)
    tr = d["traces"]
    B, traces, sp = brake_analysis(tr, wx, wet)
    D, T = tyre_analysis(tr, wx, wet)
    fig_trace(B, traces)
    fig_rank(B)
    fig_tyre(D)
    json.dump(S, open(OUT / "thermal_results.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(json.dumps(S, indent=1, default=float))
    print(B.round(2).to_string())
    print(pd.Series(sp).round(3).to_string())
    print(T.round(3).to_string())
