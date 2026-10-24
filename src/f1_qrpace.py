#!/usr/bin/env python3
"""
F1 2025: qualifying pace vs race pace, by team.

Usage:  python f1_qrpace.py <data_folder> <out_folder>     (run f1_analysis.py first: needs pace_by_round.csv)

Per dry round both pace measures are expressed as % vs the field mean of the same drivers:
  quali_pct = best qualifying lap;  race_pct = fuel/tyre-corrected race pace from the cluster-robust model.
Delta = race_pct - quali_pct: positive means the team is relatively slower in the race than over one lap.
Tests: paired t-tests over rounds per team (Holm-corrected), bootstrap CIs, Spearman rank agreement, and the link
to each team's tyre-age trend and to circuit conditions.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
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


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    out = np.empty_like(adj)
    out[o] = adj
    return out


def build_table(tr, rp, team_of_round, wet):
    q = tr.groupby(["Round", "Driver"]).LapTime_s.first().reset_index()
    q = q[~q.Round.isin(wet)]
    r = rp.rename(columns={"pace_pct": "race_pct"})[["Round", "Driver", "Team", "race_pct"]]
    D = q.merge(r, on=["Round", "Driver"], how="inner")
    D["quali_pct"] = 100 * (D.LapTime_s / D.groupby("Round").LapTime_s.transform("mean") - 1)
    D["race_pct"] = D.race_pct - D.groupby("Round").race_pct.transform("mean")
    D["quali_pct"] = D.quali_pct - D.groupby("Round").quali_pct.transform("mean")
    return D


def team_analysis(D):
    P = D.groupby(["Round", "Team"]).agg(quali_pct=("quali_pct", "mean"), race_pct=("race_pct", "mean"), n=("Driver", "size")).reset_index()
    P["delta"] = P.race_pct - P.quali_pct
    P.to_csv(TAB / "qr_team_by_round.csv", index=False)
    rng = np.random.default_rng(11)
    rows = []
    for t, g in P.groupby("Team"):
        t_ = stats.ttest_1samp(g.delta, 0)
        b = [g.delta.iloc[rng.integers(0, len(g), len(g))].mean() for _ in range(3000)]
        rows.append(dict(Team=t, rounds=len(g), quali_pct=g.quali_pct.mean(), race_pct=g.race_pct.mean(), delta=g.delta.mean(),
                         lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), t=t_.statistic, p=t_.pvalue,
                         better_in_race_rounds=int((g.delta < 0).sum())))
    T = pd.DataFrame(rows)
    T["p_holm"] = holm(T.p.to_numpy())
    T = T.sort_values("delta").reset_index(drop=True)
    T.to_csv(TAB / "qr_team_summary.csv", index=False)
    rho = stats.spearmanr(T.quali_pct, T.race_pct)
    S["teams"] = dict(rounds=int(P.Round.nunique()), teams=int(len(T)),
                      rank_spearman=float(rho.statistic), rank_p=float(rho.pvalue),
                      best_race_vs_quali=T.iloc[0].Team, best_delta=float(T.iloc[0].delta),
                      worst_race_vs_quali=T.iloc[-1].Team, worst_delta=float(T.iloc[-1].delta),
                      significant_holm=[dict(team=r.Team, delta=float(r.delta), p_holm=float(r.p_holm)) for r in T.itertuples() if r.p_holm < .05],
                      n_p05=int((T.p < .05).sum()))
    return P, T


def tyre_trend_by_team(L):
    d = L[L.Rep & ~L.Wet & L.Compound.isin(["SOFT", "MEDIUM", "HARD"]) & L.TyreLife.notna() & (L.LapNumber > 1)].copy()
    d["y"] = d.LapTime_s - d.MedLap
    d["age"] = d.TyreLife
    d["lf"] = d.LapNumber / d.N_race
    d = d[d.y.abs() < 8]
    key = ["Round", "Compound"]
    for c in ("y", "age", "lf"):
        d[c + "_d"] = d[c] - d.groupby(key)[c].transform("mean")
    rng = np.random.default_rng(2)
    rows = []
    for team, g in d.groupby("Team"):
        def slope(gg):
            X = np.column_stack([gg.age_d, gg.lf_d])
            b, *_ = np.linalg.lstsq(X, gg.y_d, rcond=None)
            return b[0]
        s0 = slope(g)
        rounds = g.Round.unique()
        bs = []
        for _ in range(200):
            pick = rng.choice(rounds, len(rounds))
            bs.append(slope(pd.concat([g[g.Round == r] for r in pick])))
        rows.append(dict(Team=team, age_slope_s_per_lap=s0, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), laps=len(g)))
    TT = pd.DataFrame(rows).sort_values("age_slope_s_per_lap")
    TT.to_csv(TAB / "qr_team_tyre_trend.csv", index=False)
    return TT


def conditions(P, wx, ev):
    # does the race-vs-quali gap of the field-level teams relate to conditions?  use per-round spread of delta
    wxi = wx.set_index("Round")
    evi = ev.set_index("Round")
    sp = P.groupby("Round").delta.agg(["std", "mean"]).rename(columns={"std": "delta_spread"})
    sp["track_c"] = wxi.loc[sp.index].TrackTemp_C
    sp["street"] = (evi.loc[sp.index].CircuitType != "Permanent").astype(int)
    r1 = stats.pearsonr(sp.track_c, sp.delta_spread)
    st = sp[sp.street == 1].delta_spread
    pm = sp[sp.street == 0].delta_spread
    w = stats.ttest_ind(st, pm, equal_var=False)
    S["conditions"] = dict(r_spread_vs_tracktemp=float(r1.statistic), p_tracktemp=float(r1.pvalue),
                           spread_street=float(st.mean()), spread_permanent=float(pm.mean()), p_street=float(w.pvalue),
                           n_street=int(len(st)), n_perm=int(len(pm)))
    sp.to_csv(TAB / "qr_conditions.csv")
    return sp


def fig_scatter(T):
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    lim = [min(T.quali_pct.min(), T.race_pct.min()) - .15, max(T.quali_pct.max(), T.race_pct.max()) + .15]
    ax.plot(lim, lim, color=MUTED, lw=1.5)
    ax.scatter(T.quali_pct, T.race_pct, s=80, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
    off = {"Red Bull Racing": (-5, 9), "Racing Bulls": (-74, 4), "Williams": (8, -14), "Mercedes": (-62, 6), "Ferrari": (8, -14)}
    for _, r in T.iterrows():
        ax.annotate(r.Team.replace(" F1 Team", "").replace(" Racing", ""), (r.quali_pct, r.race_pct), xytext=off.get(r.Team, (7, 5)),
                    textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Qualifying pace vs field mean (%; negative = faster)")
    ax.set_ylabel("Race pace vs field mean (%; negative = faster)")
    ax.text(lim[0] + .05, lim[1] - .12, "Above the line: relatively slower in the race", fontsize=8.5, color=INK2)
    ax.set_title("Qualifying pace vs race pace by team")
    savefig(fig, "fig19_quali_vs_race.png")


def fig_delta(T):
    t = T.sort_values("delta", ascending=False)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    y = np.arange(len(t))
    ax.hlines(y, t.lo, t.hi, color=GRID, lw=4, zorder=1)
    ax.scatter(t.delta, y, s=64, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(t.Team.str.replace(" F1 Team", "").str.replace(" Racing", ""))
    ax.set_xlabel("Race pace minus qualifying pace (% of lap; positive = slower in the race), 95% CI over rounds")
    ax.set_title("Which teams are better on Sunday than Saturday")
    ax.grid(axis="y", visible=False)
    savefig(fig, "fig20_race_minus_quali.png")


if __name__ == "__main__":
    d = fa.load()
    wx = d["weather"]
    wet = set(wx.loc[wx.RainFraction >= fa.WET_RAIN_FRACTION, "Round"])
    fa.R["wet_rounds"] = sorted(int(x) for x in wet)
    L = fa.tag_laps(d["laps"], d["race_control"], wx)
    rp = pd.read_csv(TAB / "pace_by_round.csv")
    D = build_table(d["traces"], rp, None, wet)
    D.to_csv(TAB / "qr_driver_by_round.csv", index=False)
    P, T = team_analysis(D)
    TT = tyre_trend_by_team(L)
    m = T.merge(TT, on="Team")
    r = stats.spearmanr(m.delta, m.age_slope_s_per_lap)
    S["tyre_link"] = dict(teams=int(len(m)), spearman=float(r.statistic), p=float(r.pvalue),
                          lowest_trend=TT.iloc[0].Team, highest_trend=TT.iloc[-1].Team)
    conditions(P, wx, d["events"])
    fig_scatter(T)
    fig_delta(T)
    S["drivers"] = dict(n=int(len(D)), corr_quali_race=float(D[["quali_pct", "race_pct"]].corr().iloc[0, 1]))
    json.dump(S, open(OUT / "qrpace_results.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(json.dumps(S, indent=1, default=float))
    print(T.round(3).to_string())
    print(TT.round(4).to_string())
