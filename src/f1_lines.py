#!/usr/bin/env python3
"""
F1 2025: racing-line and g-g analysis from qualifying position/speed traces (~255 samples per lap, one every 20 m).

Usage:  python f1_lines.py <data_folder> <out_folder>      (same folders as f1_analysis.py)

1. Resample every driver's fastest qualifying lap onto a common distance grid per circuit.
2. Find corners on the fastest lap (curvature radius < 400 m) and split every lap into corner and straight zones.
3. Decompose each lap's time against the field mean into corner time and straight time -> per team
   (downforce-heavy vs low-drag character).
4. Per-corner apex speed and minimum radius per driver; teammate comparison.
5. g-g diagram: friction-circle envelope by speed class and per-driver utilisation of it.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import signal, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
import f1_analysis as fa

OUT, TAB, FIG = fa.OUT, fa.OUT / "tables", fa.OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, MUTED, GRID, AXIS = fa.SURF, fa.INK, fa.INK2, fa.MUTED, fa.GRID, fa.AXIS
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
S = {}
R_CORNER = 400.0       # m, curvature radius below which a sample is "corner"

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


# ------------------------------------------------------------------ resampling and geometry
def resample_lap(g, n):
    """Common LapFrac grid (n points) for one lap."""
    g = g.sort_values("LapFrac").drop_duplicates("LapFrac")
    u = np.linspace(0, 1, n, endpoint=False)
    f = lambda c: np.interp(u, g.LapFrac.to_numpy(), g[c].to_numpy())
    d = dict(u=u, x=0.1 * f("X"), y=0.1 * f("Y"), v=f("Speed") / 3.6, t=f("Time_s"),
             brk=f("Brake"), thr=f("Throttle"))
    return d


def curvature(x, y):
    """Signed curvature (1/m) of a closed path sampled at ~constant spacing."""
    xs, ys = (signal.savgol_filter(a, 7, 2, mode="wrap") for a in (x, y))
    dx, dy = np.gradient(xs), np.gradient(ys)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return (dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-6), np.hypot(dx, dy)


def find_corners(kappa, v, step_m):
    """Corner zones on the reference lap: contiguous |kappa| > 1/R_CORNER, small gaps merged."""
    mask = np.abs(kappa) > 1 / R_CORNER
    n = len(mask)
    idx = np.where(mask)[0]
    zones = []
    if len(idx) == 0:
        return zones
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > 3:
            zones.append((start, prev))
            start = i
        prev = i
    zones.append((start, prev))
    zones = [(a, b) for a, b in zones if b - a >= 2]
    # drop zones with almost no speed loss (kinks, not corners)
    out = []
    for a, b in zones:
        seg = v[max(a - 6, 0):b + 7]
        if seg.max() - v[a:b + 1].min() > 8 / 3.6:
            out.append((a, b))
    return out


# ------------------------------------------------------------------ per round
def analyse_round(tr, team_of):
    best = tr.groupby("Driver").LapTime_s.first().sort_values()
    drivers = [d for d in best.index if (tr[tr.Driver == d].LapFrac.nunique() > 100)]
    ref = drivers[0]
    lap_len = float(tr[tr.Driver == ref].LapLenEst_m.iloc[0])
    n = int(round(lap_len / 20))
    step = lap_len / n
    laps = {d: resample_lap(tr[tr.Driver == d], n) for d in drivers}
    L0 = laps[ref]
    kap, ds = curvature(L0["x"], L0["y"])
    zones = find_corners(kap, L0["v"], step)
    zone_of = np.zeros(n, int)         # 0 straight, k>0 corner k
    for k, (a, b) in enumerate(zones, 1):
        zone_of[a:b + 1] = k
    rows, crow = [], []
    for k, (a, b) in enumerate(zones, 1):
        ia = a + int(np.argmin(L0["v"][a:b + 1]))
        crow.append(dict(corner=k, start_m=a * step, end_m=b * step, apex_m=ia * step, ref_apex_kph=L0["v"][ia] * 3.6,
                         ref_min_radius_m=1 / max(abs(kap[a:b + 1]).max(), 1e-6), length_m=(b - a + 1) * step,
                         direction="L" if kap[ia] > 0 else "R"))
    for d, Ld in laps.items():
        kd, dsd = curvature(Ld["x"], Ld["y"])
        t = Ld["t"]
        dt = np.diff(np.append(t, best[d]))                       # time per grid cell, closes the lap
        dt = np.clip(dt, 0, None)
        tc = dt[zone_of > 0].sum()
        ts = dt[zone_of == 0].sum()
        path = float(np.sum(dsd))                                  # driven path length (m), start-finish to start-finish
        rows.append(dict(Driver=d, Team=team_of.get(d), lap_s=float(best[d]), corner_s=tc, straight_s=ts,
                         corner_frac=float((zone_of > 0).mean()), path_m=path))
        for k, (a, b) in enumerate(zones, 1):
            seg = slice(a, b + 1)
            ia = a + int(np.argmin(Ld["v"][seg]))
            crow_ = dict(Driver=d, corner=k, apex_kph=Ld["v"][ia] * 3.6,
                         min_radius_m=1 / max(abs(kd[seg]).max(), 1e-6),
                         entry_kph=Ld["v"][max(a - 4, 0)] * 3.6, exit_kph=Ld["v"][min(b + 4, n - 1)] * 3.6,
                         zone_time_s=float(dt[seg].sum()),
                         brake_pct=float((Ld["brk"][max(a - 8, 0):b + 1] > 0.5).mean() * 100))
            rows_c.append(crow_)
    return pd.DataFrame(rows), pd.DataFrame(crow), laps, ref, zones, zone_of, step


rows_c = []


# ------------------------------------------------------------------ main analyses
def zone_decomposition(Z):
    """Time vs the field mean in the same round, split into corners and straights."""
    Z = Z.copy()
    for c in ("corner_s", "straight_s", "lap_s"):
        Z[c + "_rel"] = Z[c] - Z.groupby("Round")[c].transform("mean")
    Z.to_csv(TAB / "lines_zone_time_by_driver_round.csv", index=False)
    T = Z.groupby("Team").agg(rounds=("Round", "nunique"), corner_rel=("corner_s_rel", "mean"),
                              straight_rel=("straight_s_rel", "mean"), lap_rel=("lap_s_rel", "mean")).reset_index()
    rng = np.random.default_rng(3)
    cis = []
    for team, g in Z.groupby("Team"):
        per = g.groupby("Round")[["corner_s_rel", "straight_s_rel"]].mean()
        b = [per.iloc[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)]
        b = np.array(b)
        cis.append(dict(Team=team, corner_lo=np.percentile(b[:, 0], 2.5), corner_hi=np.percentile(b[:, 0], 97.5),
                        straight_lo=np.percentile(b[:, 1], 2.5), straight_hi=np.percentile(b[:, 1], 97.5)))
    T = T.merge(pd.DataFrame(cis), on="Team").sort_values("lap_rel")
    T.to_csv(TAB / "lines_team_corner_straight.csv", index=False)
    S["zone_split"] = dict(corner_share_of_lap_mean=float((Z.corner_s / Z.lap_s).mean()),
                           corner_frac_distance_mean=float(Z.corner_frac.mean()),
                           r_corner_vs_straight=float(stats.pearsonr(T.corner_rel, T.straight_rel).statistic),
                           most_corner_strong=T.sort_values("corner_rel").iloc[0].Team,
                           most_straight_strong=T.sort_values("straight_rel").iloc[0].Team)
    return T


def apex_tables(C, Z, ev):
    C = C.copy()
    # relative apex speed against the field mean at the same corner
    C["apex_rel"] = C.apex_kph - C.groupby(["Round", "corner"]).apex_kph.transform("mean")
    C["radius_rel"] = C.min_radius_m / C.groupby(["Round", "corner"]).min_radius_m.transform("median") - 1
    C.to_csv(TAB / "lines_corner_driver.csv", index=False)
    d = C.groupby("Driver").agg(apex_rel=("apex_rel", "mean"), radius_rel_pct=("radius_rel", lambda s: 100 * s.mean()),
                                corners=("corner", "size")).reset_index()
    team = Z.drop_duplicates("Driver").set_index("Driver").Team
    d["Team"] = d.Driver.map(team)
    d.sort_values("apex_rel", ascending=False).to_csv(TAB / "lines_driver_apex.csv", index=False)
    # teammate comparison of apex speed (paired by corner)
    rows = []
    pairs = Z.groupby("Team").Driver.unique()
    for tm, drs in pairs.items():
        drs = [x for x in drs]
        for i in range(len(drs)):
            for j in range(i + 1, len(drs)):
                a, b = C[C.Driver == drs[i]], C[C.Driver == drs[j]]
                m = a.merge(b, on=["Round", "corner"], suffixes=("_a", "_b"))
                if len(m) < 80:
                    continue
                diff = m.apex_kph_a - m.apex_kph_b
                per_round = diff.groupby(m.Round).mean()
                t = stats.ttest_1samp(per_round, 0)
                rows.append(dict(Team=tm, A=drs[i], B=drs[j], rounds=len(per_round), corners=len(m),
                                 apex_diff_kph=float(per_round.mean()), p=float(t.pvalue)))
    TM = pd.DataFrame(rows).sort_values("p")
    if len(TM):
        TM["p_holm"] = np.minimum(1, np.maximum.accumulate((len(TM) - np.arange(len(TM))) * TM.p.to_numpy()))
    TM.to_csv(TAB / "lines_teammate_apex.csv", index=False)
    S["apex"] = dict(corners=int(len(C.groupby(["Round", "corner"]))), samples=int(len(C)),
                     apex_sd_kph=float(C.apex_rel.std()), driver_spread_kph=float(d.apex_rel.max() - d.apex_rel.min()),
                     fastest_apex_driver=d.sort_values("apex_rel").iloc[-1].Driver, slowest_apex_driver=d.sort_values("apex_rel").iloc[0].Driver,
                     teammate_pairs=int(len(TM)), teammate_p05=int((TM.p < .05).sum()) if len(TM) else 0,
                     teammate_holm05=int((TM.p_holm < .05).sum()) if len(TM) else 0)
    return d, TM


def gg_analysis(tr, Z):
    K = fa.trace_kinematics(tr)
    team = Z.drop_duplicates("Driver").set_index("Driver").Team
    K["cls"] = pd.cut(K.v_kph, [0, 120, 220, 400], labels=["low (<120)", "mid (120-220)", "high (>220)"])
    K["theta"] = np.degrees(np.arctan2(K.along_g, K.alat_g))
    K["r"] = np.hypot(K.alat_g, K.along_g)
    K["tb"] = (K.theta // 10 * 10).astype(int)
    env = K.groupby(["cls", "tb"]).r.apply(lambda s: np.percentile(s, 98) if len(s) > 60 else np.nan).rename("env").reset_index()
    env["env"] = env.sort_values(["cls", "tb"]).groupby("cls").env.transform(lambda x: x.rolling(3, center=True, min_periods=1).median())
    env.to_csv(TAB / "gg_envelope.csv", index=False)
    K = K.merge(env, on=["cls", "tb"], how="left")
    K["util"] = K.r / K.env
    act = K[(K.r > 1.5) & K.util.notna()]
    u = act.groupby("Driver").util.agg(["mean", "size"]).reset_index().rename(columns={"mean": "utilisation", "size": "samples"})
    u["Team"] = u.Driver.map(team)
    rng = np.random.default_rng(5)
    ci = []
    for dname, g in act.groupby("Driver"):
        per = g.groupby("Round").util.mean()
        b = [per.iloc[rng.integers(0, len(per), len(per))].mean() for _ in range(1500)]
        ci.append(dict(Driver=dname, lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5)))
    u = u.merge(pd.DataFrame(ci), on="Driver").sort_values("utilisation", ascending=False)
    u.to_csv(TAB / "gg_utilisation_driver.csv", index=False)
    S["gg"] = dict(samples=int(len(K)), peak_total_g_p99=float(K.r.quantile(.99)),
                   env_high_lat_g=float(env[(env.cls == "high (>220)") & (env.tb.isin([-10, 0, 10]))].env.max()),
                   env_low_lat_g=float(env[(env.cls == "low (<120)") & (env.tb.isin([-10, 0, 10]))].env.max()),
                   env_brake_g=float(env[(env.cls == "high (>220)") & (env.tb.isin([-90, -80, -70]))].env.max()),
                   util_top=u.iloc[0].Driver, util_top_val=float(u.iloc[0].utilisation),
                   util_bottom=u.iloc[-1].Driver, util_bottom_val=float(u.iloc[-1].utilisation),
                   util_spread=float(u.utilisation.max() - u.utilisation.min()))
    return K, env, u


# ------------------------------------------------------------------ figures
def fig_map(tr, round_no, laps, ref, zones, zone_of, step, name, circuit):
    L = laps[ref]
    x, y, v = L["x"], L["y"], L["v"] * 3.6
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = np.concatenate([pts, np.roll(pts, -1, axis=0)], axis=1)
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    lc = LineCollection(seg, array=v, cmap=cmap, linewidth=4)
    ax.add_collection(lc)
    for k, (a, b) in enumerate(zones, 1):
        ia = a + int(np.argmin(v[a:b + 1]))
        ax.scatter(x[ia], y[ia], s=64, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(str(k), (x[ia], y[ia]), xytext=(7, 7), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(lc, ax=ax, shrink=.65, pad=.02)
    cb.set_label("Speed (km/h)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(f"{circuit}: fastest qualifying lap ({ref}), corners numbered")
    savefig(fig, name)


def fig_gg(K, env):
    k = K[K.cls == "high (>220)"]
    k2 = K[K.cls == "low (<120)"]
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.scatter(k2.alat_g.sample(min(len(k2), 5000), random_state=1), k2.along_g.sample(min(len(k2), 5000), random_state=1),
               s=4, color=ORANGE, alpha=.18, edgecolor="none", label="Below 120 km/h")
    ax.scatter(k.alat_g.sample(min(len(k), 5000), random_state=1), k.along_g.sample(min(len(k), 5000), random_state=1),
               s=4, color=BLUE, alpha=.18, edgecolor="none", label="Above 220 km/h")
    for cls, col in (("low (<120)", ORANGE), ("high (>220)", BLUE)):
        e = env[env.cls == cls].dropna().sort_values("tb")
        th = np.radians(e.tb + 5)
        ax.plot(e.env * np.cos(th), e.env * np.sin(th), color=col, lw=2)
    ax.set_xlim(0, 7)
    ax.set_ylim(-5.5, 2.5)
    ax.set_xlabel("Lateral acceleration (g)")
    ax.set_ylabel("Longitudinal acceleration (g): braking negative")
    ax.set_title("g-g diagram: the grip envelope grows with speed")
    lg = ax.legend(frameon=False, loc="lower right", markerscale=4)
    savefig(fig, "fig13_gg_diagram.png")


def fig_team(T):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    ax.errorbar(T.straight_rel, T.corner_rel, xerr=[T.straight_rel - T.straight_lo, T.straight_hi - T.straight_rel],
                yerr=[T.corner_rel - T.corner_lo, T.corner_hi - T.corner_rel], fmt="none", ecolor=GRID, elinewidth=1.5, zorder=1)
    ax.scatter(T.straight_rel, T.corner_rel, s=80, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
    for _, r in T.iterrows():
        off = {"Red Bull Racing": (-62, -14), "Racing Bulls": (8, 8), "Ferrari": (8, -14)}.get(r.Team, (7, 5))
        ax.annotate(r.Team.replace(" F1 Team", "").replace(" Racing", ""), (r.straight_rel, r.corner_rel), xytext=off,
                    textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlabel("Time on straights vs field mean (s per lap; negative = faster)")
    ax.set_ylabel("Time in corners vs field mean (s per lap; negative = faster)")
    ax.set_title("Where each team gains lap time: corners or straights")
    savefig(fig, "fig14_team_corner_straight.png")


def fig_util(u):
    u = u.sort_values("utilisation")
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    y = np.arange(len(u))
    ax.hlines(y, u.lo, u.hi, color=GRID, lw=3, zorder=1)
    ax.scatter(u.utilisation, y, s=60, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(u.Driver)
    ax.set_xlabel("Share of the grip envelope used (above 1.5 g; 95% CI over rounds)")
    ax.set_title("Friction-circle utilisation by driver")
    ax.grid(axis="y", visible=False)
    savefig(fig, "fig15_gg_utilisation.png")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    d = fa.load()
    tr = d["traces"]
    laps = d["laps"]
    wx = d["weather"]
    wet = set(wx.loc[wx.RainFraction >= fa.WET_RAIN_FRACTION, "Round"])
    fa.R["wet_rounds"] = sorted(int(x) for x in wet)
    ev = d["events"].set_index("Round")
    Zs, Cs, ex = [], [], None
    for rnd, g in tr[~tr.Round.isin(wet)].groupby("Round"):
        team_of = laps[laps.Round == rnd].drop_duplicates("Driver").set_index("Driver").Team.to_dict()
        rows_c.clear()
        try:
            Zd, crow, lp, ref, zones, zone_of, step = analyse_round(g, team_of)
        except Exception as e:
            print("round", rnd, "skipped:", e)
            continue
        Zd["Round"] = rnd
        Zd["Event"] = ev.loc[rnd, "Event"]
        C = pd.DataFrame(rows_c)
        C["Round"] = rnd
        crow["Round"] = rnd
        Zs.append(Zd)
        Cs.append(C)
        if rnd == 3 or ex is None:
            ex = (rnd, lp, ref, zones, zone_of, step, ev.loc[rnd, "Circuit"])
        if rnd == 3:
            crow.to_csv(TAB / "lines_corners_example_suzuka.csv", index=False)
    Z = pd.concat(Zs, ignore_index=True)
    C = pd.concat(Cs, ignore_index=True)
    S["rounds"] = int(Z.Round.nunique())
    T = zone_decomposition(Z)
    dA, TM = apex_tables(C, Z, ev)
    K, env, u = gg_analysis(tr[~tr.Round.isin(wet)], Z)
    rnd, lp, ref, zones, zone_of, step, circ = ex
    fig_map(tr, rnd, lp, ref, zones, zone_of, step, "fig12_racing_line_map.png", circ)
    fig_gg(K, env)
    fig_team(T)
    fig_util(u)
    json.dump(S, open(OUT / "lines_results.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(json.dumps(S, indent=1, default=float))
    print(T.round(3).to_string())
    print(dA.round(2).to_string())
    print(TM.round(3).to_string())
    print(u.round(3).to_string())
