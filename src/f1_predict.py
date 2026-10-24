#!/usr/bin/env python3
"""
F1 2025: qualifying-order, pole and race-winner prediction for a circuit, validated leave-one-round-out.

Usage:  python f1_predict.py <data_folder> <out_folder>   (run f1_analysis.py and f1_lines.py first)

For each circuit the models only see the OTHER races of the season.
Qualifying: predicted gap % = driver strength (mean gap in other rounds)
            + a circuit adjustment from the team's corner-vs-straight character and the circuit's corner share.
            Compared with a baseline that ignores the circuit.
Pole probability: Monte Carlo with the model's own out-of-sample error.
Race winner: logistic regression on (expected race pace of the driver from the other rounds, grid position),
            with grid position = predicted qualifying rank when predicting.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
import f1_analysis as fa

OUT, TAB = fa.OUT, fa.OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
S = {}
RNG = np.random.default_rng(42)


def load_inputs(d, wet):
    tr = d["traces"]
    q = tr.groupby(["Round", "Driver"]).LapTime_s.first().reset_index()
    q["quali_pct"] = 100 * (q.LapTime_s / q.groupby("Round").LapTime_s.transform("mean") - 1)
    Z = pd.read_csv(TAB / "lines_zone_time_by_driver_round.csv")
    rp = pd.read_csv(TAB / "pace_by_round.csv")[["Round", "Driver", "Team", "pace_pct"]]
    rp["race_pct"] = rp.pace_pct - rp.groupby("Round").pace_pct.transform("mean")
    res = d["results"].rename(columns={"Abbreviation": "Driver"})[["Round", "Driver", "TeamName", "GridPosition", "Position", "DNF"]]
    team = d["laps"].drop_duplicates(["Round", "Driver"])[["Round", "Driver", "Team"]]
    return q, Z, rp, res, team


def circuit_table(Z):
    z = Z.assign(cf=Z.corner_s / Z.lap_s)
    c = z.groupby("Round").agg(kappa=("cf", "mean"))
    return c


def team_character(Z, train_rounds):
    z = Z[Z.Round.isin(train_rounds)].copy()
    z["fc"] = z.corner_s / z.groupby("Round").corner_s.transform("mean") - 1
    z["fs"] = z.straight_s / z.groupby("Round").straight_s.transform("mean") - 1
    return z.groupby("Team")[["fc", "fs"]].mean()


def predict_quali(rnd, q, Z, team, circ, dry):
    train = [r for r in dry if r != rnd]
    qt = q[q.Round.isin(train)].merge(team[["Round", "Driver", "Team"]], on=["Round", "Driver"], how="left")
    drv = qt.groupby("Driver").agg(base=("quali_pct", "mean"), team=("Team", lambda s: s.mode().iloc[0]))
    kappa_bar = circ.loc[train, "kappa"].mean()
    out = drv.copy()
    out["pred_baseline"] = out.base
    if rnd in circ.index:
        ch = team_character(Z, train)
        tb = out.groupby("team").base.mean()
        # circuit adjustment: shift of the team's expected gap when the circuit's corner share differs from average
        k = circ.loc[rnd, "kappa"]
        adj = {t: 100 * ((k - kappa_bar) * (ch.loc[t, "fc"] - ch.loc[t, "fs"])) if t in ch.index else 0.0 for t in tb.index}
        out["pred_adjusted"] = out.base + out.team.map(adj)
    else:
        out["pred_adjusted"] = out.base
    return out


def spearman(a, b):
    return float(stats.spearmanr(a, b).statistic)


def run():
    d = fa.load()
    wx = d["weather"]
    wet = set(int(x) for x in wx.loc[wx.RainFraction >= fa.WET_RAIN_FRACTION, "Round"])
    fa.R["wet_rounds"] = sorted(wet)
    q, Z, rp, res, team = load_inputs(d, wet)
    ev = d["events"].set_index("Round")
    circ = circuit_table(Z)
    rounds_all = sorted(ev.index)
    dry = [r for r in rounds_all if r not in wet and r in set(q.Round)]
    dry_race = [r for r in rounds_all if r not in wet]

    # ---------------- qualifying
    rows, pole_rows, preds = [], [], {}
    resid = []
    for r in rounds_all:
        pr = predict_quali(r, q, Z, team, circ, dry)
        if r not in set(q.Round):
            preds[r] = pr
            continue
        act = q[q.Round == r].set_index("Driver").quali_pct
        common = pr.index.intersection(act.index)
        pr = pr.loc[common]
        a = act.loc[common]
        row = dict(Round=r, Event=ev.loc[r, "Event"], wet=r in wet, n=len(common),
                   rho_adj=spearman(pr.pred_adjusted, a), rho_base=spearman(pr.pred_baseline, a),
                   pole_actual=a.idxmin(), pole_pred_adj=pr.pred_adjusted.idxmin(), pole_pred_base=pr.pred_baseline.idxmin(),
                   top3_overlap_adj=len(set(pr.pred_adjusted.nsmallest(3).index) & set(a.nsmallest(3).index)),
                   top3_overlap_base=len(set(pr.pred_baseline.nsmallest(3).index) & set(a.nsmallest(3).index)),
                   mae_adj=float((pr.pred_adjusted - a).abs().mean()), mae_base=float((pr.pred_baseline - a).abs().mean()))
        rows.append(row)
        if r not in wet:
            resid.append((pr.pred_adjusted - a).to_numpy())
        preds[r] = pr
    Q = pd.DataFrame(rows)
    Q.to_csv(TAB / "predict_quali_backtest.csv", index=False)
    Qd = Q[~Q.wet]
    allres = np.concatenate(resid)
    sigma = float(1.4826 * np.median(np.abs(allres - np.median(allres))))      # robust: a few crashed/aborted laps would inflate the SD
    S_sd = float(np.std(allres))
    S["quali"] = dict(
        rounds=int(len(Qd)), sigma_pct=sigma, sd_pct=S_sd,
        rho_adj=float(Qd.rho_adj.mean()), rho_base=float(Qd.rho_base.mean()),
        pole_hits_adj=int((Qd.pole_actual == Qd.pole_pred_adj).sum()), pole_hits_base=int((Qd.pole_actual == Qd.pole_pred_base).sum()),
        top3_adj=float(Qd.top3_overlap_adj.mean()), top3_base=float(Qd.top3_overlap_base.mean()),
        mae_adj=float(Qd.mae_adj.mean()), mae_base=float(Qd.mae_base.mean()),
        adj_better_rho=int((Qd.rho_adj > Qd.rho_base).sum()), adj_worse_rho=int((Qd.rho_adj < Qd.rho_base).sum()))
    # pole probability by Monte Carlo (own out-of-sample sigma; common round shock not needed for ranking)
    pole_prob = {}
    for r, pr in preds.items():
        n = len(pr)
        sims = pr.pred_adjusted.to_numpy()[None, :] + RNG.normal(0, sigma, size=(4000, n))
        win = np.bincount(sims.argmin(axis=1), minlength=n) / 4000
        pole_prob[r] = pd.Series(win, index=pr.index)
    # calibration of pole probability: probability given to the actual pole sitter
    pp = [float(pole_prob[r].get(Q.set_index("Round").loc[r, "pole_actual"], 0)) for r in Qd.Round]
    S["quali"]["mean_prob_on_actual_pole"] = float(np.mean(pp))
    S["quali"]["pole_prob_top_hits"] = int(sum(pole_prob[r].idxmax() == Q.set_index("Round").loc[r, "pole_actual"] for r in Qd.Round))

    # ---------------- race winner
    rp2 = rp.copy()
    res2 = res.copy()
    res2["Position"] = pd.to_numeric(res2.Position, errors="coerce")
    res2["GridPosition"] = pd.to_numeric(res2.GridPosition, errors="coerce")
    base = res2.merge(rp2[["Round", "Driver", "race_pct"]], on=["Round", "Driver"], how="left")
    base["won"] = (base.Position == 1).astype(int)
    # expected pace of each driver from the OTHER dry rounds
    def other_pace(rr, drv):
        x = rp2[(rp2.Driver == drv) & (rp2.Round != rr) & (rp2.Round.isin(dry_race))].race_pct
        return x.mean() if len(x) else np.nan
    base["pace_exp"] = [other_pace(r, dv) for r, dv in zip(base.Round, base.Driver)]
    # the exported results have no grid column, so the grid is taken as the qualifying order (penalties ignored)
    q["qrank"] = q.groupby("Round").LapTime_s.rank(method="first")
    base = base.merge(q[["Round", "Driver", "qrank"]], on=["Round", "Driver"], how="left")
    base["grid"] = base.qrank
    feat = base.dropna(subset=["pace_exp", "grid"]).copy()
    wins = []
    race_prob = {}
    raw_tables = {}
    for r in dry_race:
        tr_rows = feat[(feat.Round != r) & feat.Round.isin(dry_race)]
        te = base[(base.Round == r)].dropna(subset=["pace_exp"]).copy()
        if te.empty or te.won.sum() == 0:
            continue
        # grid position of each driver at the held-out race: predicted quali rank
        if r in preds:
            rank = preds[r].pred_adjusted.rank(method="first")
            te["grid_pred"] = te.Driver.map(rank).fillna(20)
        else:
            te["grid_pred"] = te.grid.fillna(20)
        m = LogisticRegression(C=1.0).fit(tr_rows[["pace_exp", "grid"]], tr_rows.won)
        te["p_raw"] = m.predict_proba(te[["pace_exp", "grid_pred"]].rename(columns={"grid_pred": "grid"}))[:, 1]
        te["p"] = te.p_raw / te.p_raw.sum()
        raw_tables[r] = te.set_index("Driver")[["p_raw", "won"]]
        race_prob[r] = te.set_index("Driver").p
        winner = te[te.won == 1].Driver.iloc[0]
        # baseline: driver with the best expected pace
        b_pace = te.loc[te.pace_exp.idxmin(), "Driver"]
        wins.append(dict(Round=r, Event=ev.loc[r, "Event"], winner=winner, pred=te.loc[te.p.idxmax(), "Driver"],
                         p_on_winner=float(te[te.won == 1].p.iloc[0]), baseline_pace_pick=b_pace,
                         pred_top3_hit=int(winner in te.nlargest(3, "p").Driver.values),
                         pole_driver_won=int(te[te.grid == 1].won.sum()) if (te.grid == 1).any() else -1))
    # the raw probabilities are over-confident (a few cars win most races): fit one temperature on the held-out predictions
    def ll(tau):
        out = []
        for r_, t_ in raw_tables.items():
            p_ = t_.p_raw ** tau
            p_ = p_ / p_.sum()
            out.append(-np.log(max(float(p_[t_.won == 1].iloc[0]), 1e-3)))
        return float(np.mean(out))
    taus = np.linspace(0.05, 1.5, 59)
    tau = float(taus[int(np.argmin([ll(x) for x in taus]))])
    S["winner_temperature"] = dict(tau=tau, logloss_before=ll(1.0), logloss_after=ll(tau))
    for r_, t_ in raw_tables.items():
        p_ = t_.p_raw ** tau
        race_prob[r_] = p_ / p_.sum()
    for w in wins:
        pw = race_prob[w["Round"]]
        w["p_on_winner"] = float(pw.get(w["winner"], 0))
        w["pred"] = pw.idxmax()
        w["pred_top3_hit"] = int(w["winner"] in pw.nlargest(3).index)
    W = pd.DataFrame(wins)
    W.to_csv(TAB / "predict_winner_backtest.csv", index=False)
    S["winner"] = dict(rounds=int(len(W)), top1=int((W.winner == W.pred).sum()), top3=int(W.pred_top3_hit.sum()),
                       baseline_pace_pick=int((W.winner == W.baseline_pace_pick).sum()),
                       pole_won=int((W.pole_driver_won == 1).sum()), pole_known=int((W.pole_driver_won >= 0).sum()),
                       mean_p_on_winner=float(W.p_on_winner.mean()),
                       mean_logloss=float(-np.log(W.p_on_winner.clip(1e-3)).mean()),
                       uniform_logloss=float(np.log(20)))

    # ---------------- app export: leave-one-round-out predictions for every circuit
    # "actual" for display
    qa = q.set_index(["Round", "Driver"]).quali_pct
    app = {}
    for r in rounds_all:
        entry = dict(Round=int(r), Event=ev.loc[r, "Event"], wet=bool(r in wet), has_quali_model=bool(r in preds),
                     has_race_model=bool(r in race_prob))
        if r in preds:
            pr = preds[r].sort_values("pred_adjusted")
            gap = pr.pred_adjusted - pr.pred_adjusted.min()
            if r in set(q.Round):
                laps_s = float(q[q.Round == r].LapTime_s.mean())
            else:                                      # no qualifying traces: scale the race median lap by the usual quali/race ratio
                ratio = np.median([q[q.Round == k].LapTime_s.mean() / d["laps"][d["laps"].Round == k].LapTime_s.median() for k in dry])
                laps_s = float(ratio * d["laps"][d["laps"].Round == r].LapTime_s.median())
            entry["quali"] = [dict(driver=dv, team=pr.loc[dv, "team"], gap_pct=float(gap[dv]),
                                   gap_s=float(gap[dv] / 100 * laps_s), p_pole=float(pole_prob[r].get(dv, 0)))
                              for dv in pr.index[:12]]
            act = q[q.Round == r].sort_values("LapTime_s")
            entry["actual_quali_top3"] = act.Driver.head(3).tolist()
        if r in race_prob:
            p = race_prob[r].sort_values(ascending=False)
            entry["winner"] = [dict(driver=dv, p=float(p[dv])) for dv in p.index[:6]]
            wr = res2[(res2.Round == r) & (res2.Position == 1)].Driver
            entry["actual_winner"] = wr.iloc[0] if len(wr) else None
        app[int(r)] = entry
    json.dump(app, open(OUT / "app_predictions.json", "w"), indent=1, default=float, allow_nan=False)
    json.dump(S, open(OUT / "predict_results.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(json.dumps(S, indent=1, default=float))
    print(Q.round(3).to_string())
    print(W.round(3).to_string())


if __name__ == "__main__":
    run()
