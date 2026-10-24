#!/usr/bin/env python3
"""
2025 Formula 1 telemetry analysis: pace statistics, kinematics, energy and power physics.

Reads the files written by f1_openf1_export.py and writes tables, figures and
results.json:

    python f1_analysis.py <folder with the f1_2025_*.csv files> <output folder>

Physical constants and assumptions are listed at the top; everything that is an
estimate rather than a measurement is labelled as such in the outputs.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

IN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/uploads/f1_2025")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("analysis_out")
FIG, TAB = OUT / "figures", OUT / "tables"
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- constants
G = 9.80665              # m/s^2
M_CAR = 800.0            # kg, 2025 minimum weight incl. driver, excl. fuel (McLaren MCL39 spec)
FUEL0 = 100.0            # kg, ASSUMED race-start fuel (regulatory maximum is 110 kg)
M_QUALI = M_CAR + 10.0   # kg, ASSUMED qualifying-lap mass (low fuel)
MGUK_KW = 120.0          # kW, 2025 MGU-K maximum power
REC_CAP_MJ = 2.0         # MJ/lap, 2025 MGU-K maximum energy recovery
DEP_CAP_MJ = 4.0         # MJ/lap, 2025 MGU-K maximum deployment
WET_RAIN_FRACTION = 0.15 # rounds with more rain than this are excluded from pace/physics
SLOW_FACTOR = 1.07       # keep laps within 107% of the driver's median
RNG = np.random.default_rng(2025)
R = {}                   # key numbers collected for the report

# ---------------------------------------------------------------- style
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated all-pairs)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
        "figure.dpi": 110,
    })


style()


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def epoch(s):
    return (pd.to_datetime(s, utc=True, format="ISO8601") - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def ci_mean(x, level=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * se
    return m, m - h, m + h


def boot_ci(x, fn=np.mean, n=4000, level=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    b = [fn(RNG.choice(x, len(x))) for _ in range(n)]
    return tuple(np.percentile(b, [50 * (1 - level), 100 - 50 * (1 - level)]))


# ---------------------------------------------------------------- load + clean
def load():
    d = {k: pd.read_csv(IN / f"f1_2025_{k}.csv") for k in ("laps", "results", "events", "race_control", "weather")}
    d["speedbins"] = pd.read_csv(IN / "f1_2025_speedbins.csv.gz")
    d["traces"] = pd.read_csv(IN / "f1_2025_quali_traces.csv.gz")
    return d


def neutral_intervals(rc):
    """Safety-car / VSC periods per round from the race-control messages."""
    sc = rc[rc.category == "SafetyCar"].copy()
    sc["t"] = epoch(sc["date"])
    out = {}
    for rnd, g in sc.groupby("Round"):
        iv, start = [], None
        for _, r in g.sort_values("t").iterrows():
            msg = str(r["message"]).upper()
            if "DEPLOYED" in msg:
                start = r["t"] if start is None else start
            elif ("IN THIS LAP" in msg or "ENDING" in msg) and start is not None:
                iv.append((start, r["t"]))
                start = None
        if start is not None:
            iv.append((start, start + 600))
        out[rnd] = iv
    return out


def tag_laps(laps, rc, wx):
    L = laps.copy()
    L["t0"] = epoch(L["LapStart"])
    L["t1"] = L["t0"] + L["LapTime_s"]
    iv = neutral_intervals(rc)
    L["Neutral"] = [
        bool(pd.notna(a) and any(a < e and b > s for s, e in iv.get(r, [])))
        for r, a, b in zip(L["Round"], L["t0"], L["t1"])]
    L["TelOK"] = (L.TopSpeed_kph.notna() & (L.MeanSpeed_kph >= 80) & (L.Brake_pct <= 60)
                  & (L.TopSpeed_kph - L.MeanSpeed_kph > 10) & ((L.LapDur_s - L.LapTime_s).abs() <= 1.5)
                  & (L.RPM_mean > 1) & (L.BrakeZones > 0))
    wet = set(wx.loc[wx.RainFraction >= WET_RAIN_FRACTION, "Round"])
    R["wet_rounds"] = sorted(int(x) for x in wet)
    L["Wet"] = L.Round.isin(wet) | (L.Compound == "INTERMEDIATE")
    base = (~L.PitIn & ~L.PitOut & ~L.Neutral & (L.LapNumber > 1) & L.LapTime_s.notna())
    med = L[base].groupby(["Round", "Driver"])["LapTime_s"].median().rename("MedLap")
    L = L.join(med, on=["Round", "Driver"])
    L["Rep"] = base & (L.LapTime_s <= SLOW_FACTOR * L.MedLap)
    L["N_race"] = L.groupby("Round")["LapNumber"].transform("max")
    L["Mass_kg"] = M_CAR + FUEL0 * (1 - L.LapNumber / L.N_race)
    R["laps_total"] = int(len(L))
    R["laps_representative"] = int(L.Rep.sum())
    R["laps_rep_dry_telemetry"] = int((L.Rep & ~L.Wet & L.TelOK).sum())
    R["laps_neutralised"] = int(L.Neutral.sum())
    return L


# ---------------------------------------------------------------- pace model
def fit_round_pace(d):
    """
    Lap time = driver effect + compound effect + tyre-age slope per compound + fuel
    (lap number) slope. Cluster-robust (driver-stint) covariance.
    """
    cnt = d.groupby("Driver").size()
    drivers = sorted(cnt[cnt >= 10].index)
    d = d[d.Driver.isin(drivers)]
    if len(drivers) < 4:
        return None
    cols, names = [], []
    for drv in drivers:
        cols.append((d.Driver == drv).astype(float).to_numpy())
        names.append(f"D:{drv}")
    comps = [c for c in ("SOFT", "MEDIUM", "HARD") if (d.Compound == c).sum() >= 25]
    base = max(comps, key=lambda c: (d.Compound == c).sum()) if comps else None
    for c in comps:
        if c != base:
            cols.append((d.Compound == c).astype(float).to_numpy())
            names.append(f"C:{c}")
    for c in comps:
        cols.append(((d.Compound == c) * d.TyreLife).astype(float).to_numpy())
        names.append(f"T:{c}")
    cols.append(d.LapNumber.astype(float).to_numpy())
    names.append("FUEL")
    X = np.column_stack(cols)
    y = d.LapTime_s.to_numpy()
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    grp = (d.Driver + "_" + d.Stint.astype(str)).to_numpy()
    meat = np.zeros((X.shape[1],) * 2)
    for g in np.unique(grp):
        s = X[grp == g].T @ e[grp == g]
        meat += np.outer(s, s)
    Gn, n, k = len(np.unique(grp)), len(y), np.linalg.matrix_rank(X)
    V = XtXi @ meat @ XtXi * (Gn / max(Gn - 1, 1)) * ((n - 1) / max(n - k, 1))
    D = len(drivers)
    C = np.eye(D) - np.ones((D, D)) / D
    delta = C @ beta[:D]
    Vd = C @ V[:D, :D] @ C.T
    out = dict(drivers=drivers, delta=delta, Vd=Vd, n=cnt[drivers].to_numpy(),
               med=float(np.median(y)), resid_sd=float(e.std()),
               fuel=float(beta[names.index("FUEL")]), fuel_se=float(np.sqrt(V[names.index("FUEL"), names.index("FUEL")])))
    for c in comps:
        j = names.index(f"T:{c}")
        out[f"tyre_{c}"] = float(beta[j])
        out[f"tyre_{c}_se"] = float(np.sqrt(V[j, j]))
    return out


def pace_analysis(L):
    D = L[L.Rep & ~L.Wet & L.Compound.isin(["SOFT", "MEDIUM", "HARD"]) & L.TyreLife.notna()]
    fits = {}
    for rnd, d in D.groupby("Round"):
        f = fit_round_pace(d)
        if f:
            fits[rnd] = f
    R["pace_rounds"] = len(fits)
    team_of = L.groupby(["Round", "Driver"])["Team"].agg(lambda s: s.mode().iloc[0])

    rows, rr = [], []
    for rnd, f in fits.items():
        se = np.sqrt(np.diag(f["Vd"]))
        for drv, dl, s, n in zip(f["drivers"], f["delta"], se, f["n"]):
            rows.append(dict(Round=rnd, Driver=drv, Team=team_of.get((rnd, drv)), delta_s=dl, se_s=s,
                             pace_pct=100 * dl / f["med"], laps=int(n)))
        rr.append(dict(Round=rnd, fuel_s_per_lap=f["fuel"], fuel_se=f["fuel_se"], resid_sd_s=f["resid_sd"],
                       **{k: v for k, v in f.items() if k.startswith("tyre_")}))
    P = pd.DataFrame(rows)
    RM = pd.DataFrame(rr)
    P.to_csv(TAB / "pace_by_round.csv", index=False)
    RM.to_csv(TAB / "pace_model_by_round.csv", index=False)

    # season driver ranking: mean of per-round % deficit, t interval across rounds
    drv = []
    for name, g in P.groupby("Driver"):
        if len(g) < 6:
            continue
        m, lo, hi = ci_mean(g.pace_pct)
        drv.append(dict(Driver=name, Team=g.Team.mode().iloc[0], rounds=len(g), pace_pct=m, lo=lo, hi=hi,
                        pace_s=g.delta_s.mean()))
    DR = pd.DataFrame(drv).sort_values("pace_pct").reset_index(drop=True)
    DR.to_csv(TAB / "driver_pace.csv", index=False)

    tm = P.groupby(["Round", "Team"])["pace_pct"].mean().reset_index()
    tm = tm.groupby("Team")["pace_pct"].agg(["mean", "std", "count"]).reset_index()
    tm["se"] = tm["std"] / np.sqrt(tm["count"])
    tm = tm[tm["count"] >= 6].sort_values("mean")
    tm.to_csv(TAB / "team_pace.csv", index=False)

    # teammate head-to-head (paired across rounds)
    h = []
    for rnd, f in fits.items():
        idx = {d_: i for i, d_ in enumerate(f["drivers"])}
        byteam = {}
        for d_ in f["drivers"]:
            byteam.setdefault(team_of.get((rnd, d_)), []).append(d_)
        for team, ds in byteam.items():
            if len(ds) == 2:
                a, b = sorted(ds)
                i, j = idx[a], idx[b]
                diff = f["delta"][i] - f["delta"][j]            # >0: a slower
                var = f["Vd"][i, i] + f["Vd"][j, j] - 2 * f["Vd"][i, j]
                h.append(dict(Round=rnd, Team=team, A=a, B=b, diff_s=diff, se_s=np.sqrt(max(var, 0)),
                              diff_pct=100 * diff / f["med"]))
    H = pd.DataFrame(h)
    H.to_csv(TAB / "teammate_by_round.csv", index=False)
    res = []
    for (a, b), g in H.groupby(["A", "B"]):
        if len(g) < 4:
            continue
        x = g.diff_s.to_numpy()
        t, p = stats.ttest_1samp(x, 0.0)
        try:
            w = stats.wilcoxon(x).pvalue
        except ValueError:
            w = np.nan
        lo, hi = boot_ci(x)
        res.append(dict(A=a, B=b, Team=g.Team.mode().iloc[0], rounds=len(x), mean_diff_s=x.mean(), lo=lo, hi=hi,
                        t=t, p_t=p, p_wilcoxon=w, dz=x.mean() / x.std(ddof=1), a_faster_rounds=int((x < 0).sum())))
    HT = pd.DataFrame(res).sort_values("p_t").reset_index(drop=True)
    p = HT.p_t.to_numpy()                                   # Holm-Bonferroni adjustment
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    HT["p_holm"] = adj
    HT.to_csv(TAB / "teammate_h2h.csv", index=False)

    # fuel and tyre coefficients across rounds
    m, lo, hi = ci_mean(RM.fuel_s_per_lap)
    t, p = stats.ttest_1samp(RM.fuel_s_per_lap.dropna(), 0.0)
    R["fuel_laptime_s_per_lap"] = dict(mean=m, lo=lo, hi=hi, p=p, rounds=int(RM.fuel_s_per_lap.notna().sum()))
    for c in ("SOFT", "MEDIUM", "HARD"):
        col = f"tyre_{c}"
        if col in RM:
            mm, l_, h_ = ci_mean(RM[col])
            R[f"tyre_deg_{c}"] = dict(mean=mm, lo=l_, hi=h_, rounds=int(RM[col].notna().sum()))
    R["pace_resid_sd_s"] = float(RM.resid_sd_s.median())
    R["top_drivers"] = DR.head(3)[["Driver", "pace_pct"]].to_dict("records")
    R["teammate_pairs_tested"] = int(len(HT))
    R["teammate_pairs_p05"] = int((HT.p_t < 0.05).sum())
    R["teammate_pairs_holm05"] = int((HT.p_holm < 0.05).sum())

    # figure: driver pace
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    y = np.arange(len(DR))[::-1]
    ax.hlines(y, DR.lo, DR.hi, color=AXIS, lw=2, zorder=1)
    ax.scatter(DR.pace_pct, y, s=46, color=C1, edgecolor=SURF, linewidth=2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d_}  ({t_})" for d_, t_ in zip(DR.Driver, DR.Team)], fontsize=8.5, color=INK2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Pace relative to the field average, % of lap time (negative = faster)")
    ax.set_title("Season race pace after tyre and fuel correction")
    ax.grid(axis="y", visible=False)
    save(fig, "fig1_driver_pace.png")

    # figure: teammate forest plot
    HT2 = HT.sort_values("mean_diff_s")
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(HT2) + 1.4))
    y = np.arange(len(HT2))[::-1]
    ax.hlines(y, HT2.lo, HT2.hi, color=AXIS, lw=2)
    sig = HT2.p_t < 0.05
    ax.scatter(HT2.mean_diff_s[sig], y[sig.to_numpy()], s=46, color=C2, edgecolor=SURF, linewidth=2, label="p < 0.05", zorder=3)
    ax.scatter(HT2.mean_diff_s[~sig], y[(~sig).to_numpy()], s=46, color=C1, edgecolor=SURF, linewidth=2, label="not significant", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} vs {b}  (n={n})" for a, b, n in zip(HT2.A, HT2.B, HT2.rounds)], fontsize=8.5, color=INK2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Mean lap-time difference, first driver minus second (s/lap; negative = first is faster)")
    ax.set_title("Teammate head-to-head, same car, paired by race")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_teammates.png")
    return P, RM, DR, HT


# ---------------------------------------------------------------- energy per circuit
def energy_analysis(L, ev):
    """
    Per-lap kinetic-energy flows against the MGU-K limits (120 kW, 2 MJ recovery,
    4 MJ deployment per lap).  Energy = (per-kg value from telemetry) x car mass.
    """
    D = L[L.Rep & ~L.Wet & L.TelOK].copy()
    D["Ebrake_MJ"] = D.BrakeEnergy_kJkg * D.Mass_kg / 1000
    D["KEgain_MJ"] = D.KEGain_kJkg * D.Mass_kg / 1000
    D["FTtime_s"] = D.FullThrottle_pct * D.LapDur_s / 100
    g = D.groupby("Round").agg(
        laps=("Round", "size"), lap_s=("LapTime_s", "median"), top_kph=("TopSpeed_kph", "median"),
        zones=("BrakeZones", "median"), brake_s=("BrakeTime_s", "median"), coast_s=("CoastTime_s", "median"),
        ft_s=("FTtime_s", "median"), Ebrake_MJ=("Ebrake_MJ", "median"), KEgain_MJ=("KEgain_MJ", "median"),
        rpm_mean=("RPM_mean", "median"), rpm_max=("RPM_max", "median"))
    g = g.join(ev.set_index("Round")[["Event", "Circuit", "CircuitType"]])
    g["Erec_power_MJ"] = MGUK_KW / 1000 * g.brake_s
    g["Erec_max_MJ"] = np.minimum(REC_CAP_MJ, g.Erec_power_MJ)
    g["limit"] = np.where(g.Erec_power_MJ < REC_CAP_MJ, "power (120 kW)", "energy cap (2 MJ)")
    g["deploy_full_power_s"] = DEP_CAP_MJ / (MGUK_KW / 1000)
    g["deploy_share_of_FT"] = np.minimum(1, g.deploy_full_power_s / g.ft_s)
    g["rec_fraction_of_braking_dKE"] = g.Erec_max_MJ / g.Ebrake_MJ
    g = g.reset_index()
    g.to_csv(TAB / "circuit_energy.csv", index=False)

    R["energy"] = dict(
        rounds=int(len(g)), Ebrake_MJ_median=float(g.Ebrake_MJ.median()), Ebrake_MJ_min=float(g.Ebrake_MJ.min()),
        Ebrake_MJ_max=float(g.Ebrake_MJ.max()),
        Ebrake_max_event=g.loc[g.Ebrake_MJ.idxmax(), "Event"], Ebrake_min_event=g.loc[g.Ebrake_MJ.idxmin(), "Event"],
        ratio_to_cap_median=float((g.Ebrake_MJ / REC_CAP_MJ).median()),
        power_limited_rounds=int((g.limit.str.startswith("power")).sum()),
        cap_limited_rounds=int((g.limit.str.startswith("energy")).sum()),
        brake_time_s_median=float(g.brake_s.median()), ft_time_s_median=float(g.ft_s.median()),
        deploy_full_power_s=DEP_CAP_MJ / (MGUK_KW / 1000), rec_full_power_s=REC_CAP_MJ / (MGUK_KW / 1000),
        KEgain_over_Ebrake=float((g.KEgain_MJ / g.Ebrake_MJ).median()))
    R["energy"]["power_limited_events"] = g.loc[g.limit.str.startswith("power"), "Event"].tolist()
    R["energy"]["cap_limited_events"] = g.loc[g.limit.str.startswith("energy"), "Event"].tolist()

    # circuit type comparison (per-round medians)
    types = {k: v.Ebrake_MJ.to_numpy() for k, v in g.groupby("CircuitType") if len(v) >= 3}
    R["circuit_type_counts"] = {k: int(len(v)) for k, v in g.groupby("CircuitType")}
    R["circuit_type_Ebrake_median"] = {k: float(np.median(v)) for k, v in types.items()}
    street = [k for k in types if "Street" in k]
    perm = [k for k in types if k == "Permanent"]
    if street and perm:
        a, b = types[street[0]], types[perm[0]]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        u = stats.mannwhitneyu(a, b).pvalue
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        R["street_vs_permanent"] = dict(street_n=len(a), perm_n=len(b), street_mean=float(a.mean()),
                                        perm_mean=float(b.mean()), welch_t=float(t), welch_p=float(p),
                                        mannwhitney_p=float(u), cohen_d=float((a.mean() - b.mean()) / sp))
    # what sets braking energy?  KE ~ v^2: regress on top speed^2
    x, y = g.top_kph.to_numpy(), g.Ebrake_MJ.to_numpy()
    r = stats.pearsonr(x, y)
    R["Ebrake_vs_topspeed"] = dict(r=float(r.statistic), p=float(r.pvalue), n=int(len(g)))
    r2 = stats.pearsonr(g.zones.to_numpy(), y)
    R["Ebrake_vs_zones"] = dict(r=float(r2.statistic), p=float(r2.pvalue))
    return g


# ---------------------------------------------------------------- drag, power, DRS, fuel (speed bins)
def wls_line(x, y, w):
    """Weighted least squares y = A x + B; returns A, B, se_A, R^2."""
    W = np.sqrt(w)
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    res = (y - X @ beta) * W
    dof = max(len(x) - 2, 1)
    cov = np.linalg.pinv((X * W[:, None]).T @ (X * W[:, None])) * (res @ res) / dof
    yb = np.average(y, weights=w)
    r2 = 1 - np.sum(w * (y - X @ beta) ** 2) / np.sum(w * (y - yb) ** 2)
    return beta[0], beta[1], float(np.sqrt(cov[0, 0])), float(r2)


def wavg(x, w):
    return float(np.average(x, weights=w))


def speedbin_physics(sb, wx, L, ev):
    """
    Full-throttle acceleration curve, kinetic power, DRS drag change, MGU-K torque
    and a fuel-mass estimate, all from the speed-binned acceleration tables.

    Not done on purpose: a coast-down drag fit.  Samples with no throttle and no brake
    flag still contain light braking (the brake flag lags the real onset), energy
    harvesting and engine braking, so deceleration there is not aerodynamic drag.
    """
    wet = set(R["wet_rounds"])
    S = sb[~sb.Round.isin(wet)].copy()
    S["v"] = (S.SpeedBin_kph + 5) / 3.6
    S["a"] = S.sum_a_g / S.n * G                                # m/s^2, mean in the bin
    S["mass"] = M_CAR + FUEL0 * (1 - (S.Phase + 0.5) / 3)       # kg, assumed linear fuel burn
    S["rho"] = S.Round.map(wx.set_index("Round").AirDensity_kgm3)
    S["Pkin_kW"] = S.mass * S.a * S.v / 1000                    # power going into kinetic energy

    # ---- full-throttle curve, DRS closed
    ft = S[(S.Mode == "FT") & (S.DRS == 0)]
    c = ft.groupby("SpeedBin_kph").apply(lambda t: pd.Series(dict(
        n=t.n.sum(), a_g=wavg(t.a, t.n) / G, Pkin_kW=wavg(t.Pkin_kW, t.n),
        sd_g=np.sqrt(max(t.sum_a2_g.sum() / t.n.sum() - (t.sum_a_g.sum() / t.n.sum()) ** 2, 0)))),
        include_groups=False).reset_index()
    c = c[c.n >= 3000]
    c.to_csv(TAB / "fullthrottle_curve.csv", index=False)
    byr = ft.groupby(["Round", "SpeedBin_kph"]).apply(lambda t: pd.Series(dict(
        n=t.n.sum(), a_g=wavg(t.a, t.n) / G, Pkin_kW=wavg(t.Pkin_kW, t.n))), include_groups=False).reset_index()
    byr = byr[byr.n >= 500]
    byr.to_csv(TAB / "fullthrottle_by_round_bin.csv", index=False)
    pk = c.loc[c.Pkin_kW.idxmax()]
    term = c[c.a_g < 0.01]
    R["fullthrottle"] = dict(
        a_g_at_150=float(c.loc[c.SpeedBin_kph == 150, "a_g"].iloc[0]), a_g_at_200=float(c.loc[c.SpeedBin_kph == 200, "a_g"].iloc[0]),
        a_g_at_250=float(c.loc[c.SpeedBin_kph == 250, "a_g"].iloc[0]), a_g_at_300=float(c.loc[c.SpeedBin_kph == 300, "a_g"].iloc[0]),
        Pkin_at_200_kW=float(c.loc[c.SpeedBin_kph == 200, "Pkin_kW"].iloc[0]),
        Pkin_at_250_kW=float(c.loc[c.SpeedBin_kph == 250, "Pkin_kW"].iloc[0]),
        Pkin_at_300_kW=float(c.loc[c.SpeedBin_kph == 300, "Pkin_kW"].iloc[0]),
        Pkin_peak_kW=float(pk.Pkin_kW), Pkin_peak_at_kph=int(pk.SpeedBin_kph) + 5,
        speed_where_a_below_0p01g_kph=int(term.SpeedBin_kph.min()) + 5 if len(term) else None)

    # ---- MGU-K torque at its 120 kW limit, for the engine speeds seen at full throttle
    r_ = S[(S.Mode == "FT") & (S.SpeedBin_kph >= 100)].copy()
    r_["rpm"] = r_.sum_rpm / r_.n
    rb = r_.groupby("SpeedBin_kph").apply(lambda t: wavg(t.rpm, t.n), include_groups=False)
    tq = MGUK_KW * 1000 / (rb * 2 * np.pi / 60)
    R["mguk"] = dict(rpm_min=float(rb.min()), rpm_max=float(rb.max()), torque_at_120kW_min_Nm=float(tq.min()),
                     torque_at_120kW_max_Nm=float(tq.max()))
    pd.DataFrame(dict(rpm=rb, mguk_torque_Nm_at_120kW=tq)).to_csv(TAB / "mguk_torque_vs_speed.csv")

    # ---- DRS: same engine, less drag -> more acceleration at the same speed
    top = L[L.Rep & ~L.Wet & L.TelOK].groupby("Round").TopSpeed_kph.median()
    rows = []
    for rnd, g in S[(S.Mode == "FT") & (S.SpeedBin_kph >= 240)].groupby("Round"):
        agg = lambda t: t.groupby("SpeedBin_kph").apply(lambda u: pd.Series(dict(
            n=u.n.sum(), a=wavg(u.a, u.n), m=wavg(u.mass, u.n), v=u.v.iloc[0])), include_groups=False)
        j = agg(g[g.DRS == 1]).join(agg(g[g.DRS == 0]), lsuffix="_o", rsuffix="_c", how="inner")
        j = j[(j.n_o >= 150) & (j.n_c >= 150)]
        if len(j) < 2:
            continue
        w = np.minimum(j.n_o, j.n_c)
        m_ = (j.m_o + j.m_c) / 2
        dF = m_ * (j.a_o - j.a_c)
        dCdA = 2 * dF / (float(g.rho.iloc[0]) * j.v_o ** 2)
        rows.append(dict(Round=rnd, dA_ms2=wavg(j.a_o - j.a_c, w), dF_kN=wavg(dF, w) / 1000, dCdA_m2=wavg(dCdA, w),
                         bins=len(j), top_kph=float(top.get(rnd, np.nan))))
    DRS = pd.DataFrame(rows).merge(ev[["Round", "Event"]], on="Round")
    DRS.to_csv(TAB / "drs_effect_by_round.csv", index=False)
    m, lo, hi = ci_mean(DRS.dCdA_m2)
    mf, lof, hif = ci_mean(DRS.dF_kN)
    rr = stats.pearsonr(DRS.top_kph, DRS.dCdA_m2)
    R["drs"] = dict(rounds=int(len(DRS)), dCdA_m2=m, lo=lo, hi=hi, dF_kN=mf, dF_lo=lof, dF_hi=hif,
                    dA_ms2_mean=float(DRS.dA_ms2.mean()), r_vs_topspeed=float(rr.statistic), p_vs_topspeed=float(rr.pvalue),
                    lowest=DRS.loc[DRS.dCdA_m2.idxmin(), ["Event", "dCdA_m2"]].to_dict(),
                    highest=DRS.loc[DRS.dCdA_m2.idxmax(), ["Event", "dCdA_m2"]].to_dict())

    # ---- fuel mass: with power and drag unchanged, m*a is constant, so m2/m0 = a0/a2.
    # Burn is linear in laps and phases 0 and 2 are centred 2/3 of the race apart, with phase 0
    # centred when 1/6 of the fuel is gone:  F*(2/3) = (800 + F*5/6) * (1 - a0/a2)  ->  solve for F.
    fb = []
    ftd = S[(S.Mode == "FT") & (S.DRS == 0) & (S.SpeedBin_kph >= 200) & (S.SpeedBin_kph < 290)]
    for (rnd, drv), g in ftd.groupby(["Round", "Driver"]):
        p0, p2 = g[g.Phase == 0].set_index("SpeedBin_kph"), g[g.Phase == 2].set_index("SpeedBin_kph")
        j = p0[["n", "a"]].join(p2[["n", "a"]], lsuffix="0", rsuffix="2", how="inner")
        j = j[(j.n0 >= 40) & (j.n2 >= 40)]
        if len(j) < 3:
            continue
        w = np.minimum(j.n0, j.n2)
        a0, a2 = wavg(j.a0, w), wavg(j.a2, w)
        if a0 <= 0 or a2 <= 0:
            continue
        q = 1 - a0 / a2
        den = 2 / 3 - 5 * q / 6
        if den <= 0:
            continue
        F = M_CAR * q / den
        n_race = float(L[L.Round == rnd].LapNumber.max())
        fb.append(dict(Round=rnd, Driver=drv, a0=a0, a2=a2, q=q, race_fuel_kg=F, burn_kg_per_lap=F / n_race, n_race=n_race))
    FB = pd.DataFrame(fb)
    FB.to_csv(TAB / "fuel_burn_estimates.csv", index=False)
    per_round = FB.groupby("Round").race_fuel_kg.median()
    med = float(FB.race_fuel_kg.median())
    lo_, hi_ = boot_ci(FB.race_fuel_kg.to_numpy(), np.median)
    burn = float(FB.burn_kg_per_lap.median())
    R["fuel"] = dict(n_estimates=int(len(FB)), median_race_fuel_kg=med, ci_lo=lo_, ci_hi=hi_,
                     iqr_lo=float(FB.race_fuel_kg.quantile(.25)), iqr_hi=float(FB.race_fuel_kg.quantile(.75)),
                     per_round_median_min=float(per_round.min()), per_round_median_max=float(per_round.max()),
                     burn_kg_per_lap_median=burn,
                     rounds_with_estimate=int(per_round.size),
                     seconds_per_kg=float(abs(R["fuel_laptime_s_per_lap"]["mean"]) / burn))
    return S, c, byr, DRS, FB


# ---------------------------------------------------------------- lateral grip from qualifying traces
def trace_kinematics(tr):
    """Longitudinal and lateral acceleration along each fastest qualifying lap."""
    wet = set(R["wet_rounds"])
    out = []
    for (rnd, drv), g in tr[~tr.Round.isin(wet)].groupby(["Round", "Driver"]):
        g = g.sort_values("Distance_m")
        if len(g) < 80:
            continue
        v = g.Speed.to_numpy() / 3.6
        t = g.Time_s.to_numpy()
        x, y = 0.1 * g.X.to_numpy(), 0.1 * g.Y.to_numpy()          # position units are decimetres
        xs, ys = signal.savgol_filter(x, 7, 2), signal.savgol_filter(y, 7, 2)
        dx, dy = np.gradient(xs), np.gradient(ys)
        ds = np.maximum(np.hypot(dx, dy), 1.0)
        kappa = signal.savgol_filter(np.gradient(np.unwrap(np.arctan2(dy, dx))), 5, 2) / ds
        vs = signal.savgol_filter(v, 5, 2)
        d = pd.DataFrame(dict(Round=rnd, Driver=drv, v_kph=vs * 3.6, alat_g=vs ** 2 * np.abs(kappa) / G,
                              along_g=np.gradient(vs, t) / G, brake=g.Brake.to_numpy(), thr=g.Throttle.to_numpy()))
        out.append(d.iloc[4:-4])
    K = pd.concat(out, ignore_index=True)
    return K[(K.v_kph > 55) & (K.alat_g < 7) & (K.along_g.abs() < 6)]


def lateral_analysis(tr, wx, ev):
    K = trace_kinematics(tr)
    K.sample(min(len(K), 60000), random_state=1).to_csv(TAB / "trace_kinematics_sample.csv", index=False)
    rho = wx.set_index("Round").AirDensity_kgm3
    rows = []
    for rnd, g in K.groupby("Round"):
        g = g.assign(bin=(g.v_kph // 20 * 20).astype(int))
        e = g.groupby("bin").agg(n=("alat_g", "size"), p97=("alat_g", lambda s: np.percentile(s, 97))).reset_index()
        e = e[(e.n >= 40) & (e.bin >= 100) & (e.bin <= 300)]
        if len(e) < 5:
            continue
        v2 = ((e.bin + 10) / 3.6) ** 2
        beta, mu, se, r2 = wls_line(v2.to_numpy(), e.p97.to_numpy(), e.n.to_numpy())
        if mu <= 0 or beta <= 0:
            continue
        rows.append(dict(Round=rnd, mu=mu, beta=beta, r2=r2, CLA_m2=2 * beta * M_QUALI * G / (mu * float(rho[rnd])), bins=len(e)))
    E = pd.DataFrame(rows).merge(ev[["Round", "Event"]], on="Round")
    top = tr.groupby("Round").Speed.max()
    E["top_kph"] = E.Round.map(top)
    E.to_csv(TAB / "lateral_grip_by_round.csv", index=False)
    ok = E[E.r2 > 0.7]
    R["lateral"] = dict(
        samples=int(len(K)), p99_alat_g=float(np.percentile(K.alat_g, 99)), max_alat_p999_g=float(np.percentile(K.alat_g, 99.9)),
        p99_brake_decel_g=float(-np.percentile(K.along_g[K.brake == 1], 1)), p99_accel_g=float(np.percentile(K.along_g, 99)),
        rounds_fitted=int(len(E)), rounds_good_fit=int(len(ok)), r2_median=float(E.r2.median()),
        mu_median=float(ok.mu.median()) if len(ok) else None, mu_min=float(ok.mu.min()) if len(ok) else None,
        mu_max=float(ok.mu.max()) if len(ok) else None,
        CLA_median_m2=float(ok.CLA_m2.median()) if len(ok) else None, CLA_min=float(ok.CLA_m2.min()) if len(ok) else None,
        CLA_max=float(ok.CLA_m2.max()) if len(ok) else None,
        CLA_min_event=ok.loc[ok.CLA_m2.idxmin(), "Event"] if len(ok) else None,
        CLA_max_event=ok.loc[ok.CLA_m2.idxmax(), "Event"] if len(ok) else None)
    if len(ok) > 4:
        rr = stats.pearsonr(ok.top_kph, ok.CLA_m2)
        R["lateral"]["CLA_vs_topspeed_r"] = float(rr.statistic)
        R["lateral"]["CLA_vs_topspeed_p"] = float(rr.pvalue)
    # ---- pooled envelope fit (per-round fits are too noisy: ~4 laps/driver/round) over 80-260 km/h
    rho_bar = float(rho.loc[sorted(set(K.Round))].mean())
    def pooled(Kx):
        g = Kx.assign(bin=(Kx.v_kph // 20 * 20).astype(int))
        e = g.groupby("bin").agg(n=("alat_g", "size"), p97=("alat_g", lambda s: np.percentile(s, 97))).reset_index()
        e = e[(e.bin >= 80) & (e.bin <= 260) & (e.n >= 100)]
        v2 = ((e.bin + 10) / 3.6) ** 2
        b, mu, se, r2 = wls_line(v2.to_numpy(), e.p97.to_numpy(), e.n.to_numpy())
        return b, mu, r2, e
    b, mu, r2, env = pooled(K)
    env.to_csv(TAB / "lateral_envelope_pooled.csv", index=False)
    rng = np.random.default_rng(7)
    rounds = np.array(sorted(set(K.Round)))
    byrd = {r: K[K.Round == r] for r in rounds}
    bs = []
    for _ in range(300):
        pick = rng.choice(rounds, len(rounds))
        bb, mm, _, _ = pooled(pd.concat([byrd[r] for r in pick]))
        bs.append((mm, 2 * bb * M_QUALI * G / (mm * rho_bar), mm * G ** 0 * 0 + bb * (100 / 3.6) ** 2 / mm))
    bs = np.array(bs)
    CLA = 2 * b * M_QUALI * G / (mu * rho_bar)
    R["lateral_pooled"] = dict(
        mu_g=float(mu), mu_ci=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
        CLA_m2=float(CLA), CLA_ci=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
        r2=float(r2), downforce_to_weight_at_250kph=float(0.5 * rho_bar * CLA * (250 / 3.6) ** 2 / (M_QUALI * G)),
        grip_at_100_g=float(mu + b * (100 / 3.6) ** 2), grip_at_250_g=float(mu + b * (250 / 3.6) ** 2),
        rho=rho_bar, rounds=int(len(rounds)))
    return K, E


if __name__ == "__main__":
    d = load()
    L = tag_laps(d["laps"], d["race_control"], d["weather"])
    P, RM, DR, HT = pace_analysis(L)
    EN = energy_analysis(L, d["events"])
    S, FT, FTR, DRS, FB = speedbin_physics(d["speedbins"], d["weather"], L, d["events"])
    K, LAT = lateral_analysis(d["traces"], d["weather"], d["events"])
    json.dump(R, open(OUT / "results.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(json.dumps({k: R[k] for k in ("fullthrottle", "mguk", "drs", "fuel", "lateral", "fuel_laptime_s_per_lap")}, indent=1, default=float))
    print(DRS.round(3).to_string())
    print(LAT.round(3).to_string())
    print(FB.describe().round(3).to_string())
