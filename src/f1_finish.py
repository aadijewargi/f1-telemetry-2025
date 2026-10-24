#!/usr/bin/env python3
"""
F1 2025: race finishing-order model, validated leave-one-round-out.

Usage:  python f1_finish.py <data_folder> <out_folder>   (run f1_analysis.py, f1_lines.py, f1_predict.py first)

For each dry race the model only sees the OTHER dry races.
Features: the driver's expected race pace (mean in the other races) and grid position (predicted qualifying rank when predicting).
Finishing position = linear fit on those two, plus Gaussian noise with the held-out residual spread;
a driver retires with probability equal to their shrunk retirement rate in the other races (then finishes behind all finishers).
Monte Carlo over the field gives expected position, P(podium), P(top 10) and P(retire).
Baselines: grid only, pace only.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
import f1_analysis as fa
import f1_predict as fp

OUT, TAB = fa.OUT, fa.OUT / "tables"
RNG = np.random.default_rng(7)
NSIM = 4000


def fit_line(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b


def simulate(mu, p_dnf, s, rng=RNG):
    n = len(mu)
    score = mu[None, :] + rng.normal(0, s, size=(NSIM, n))
    dnf = rng.random((NSIM, n)) < p_dnf[None, :]
    score = np.where(dnf, 1e3 + rng.random((NSIM, n)), score)     # retirements go behind every finisher
    pos = score.argsort(axis=1).argsort(axis=1) + 1
    return pos, dnf


def run():
    d = fa.load()
    wx = d["weather"]
    wet = set(int(x) for x in wx.loc[wx.RainFraction >= fa.WET_RAIN_FRACTION, "Round"])
    fa.R["wet_rounds"] = sorted(wet)
    q, Z, rp, res, team = fp.load_inputs(d, wet)
    ev = d["events"].set_index("Round")
    circ = fp.circuit_table(Z)
    rounds_all = sorted(ev.index)
    dry = [r for r in rounds_all if r not in wet and r in set(q.Round)]
    dry_race = [r for r in rounds_all if r not in wet]

    res = res.copy()
    res["Position"] = pd.to_numeric(res.Position, errors="coerce")
    res["DNF"] = res.DNF.astype(str).str.lower().isin(["true", "1", "yes"])
    q["qrank"] = q.groupby("Round").LapTime_s.rank(method="first")
    base = res.merge(rp[["Round", "Driver", "race_pct"]], on=["Round", "Driver"], how="left").merge(q[["Round", "Driver", "qrank"]], on=["Round", "Driver"], how="left")
    pace_by = {r: rp[rp.Round == r].set_index("Driver").race_pct for r in rounds_all}
    # a driver's expected pace from the other dry races
    def pace_exp(rr, drv):
        v = [pace_by[k].get(drv, np.nan) for k in dry_race if k != rr]
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else np.nan
    base["pace_exp"] = [pace_exp(r, dv) for r, dv in zip(base.Round, base.Driver)]
    base["pace_rank"] = base.groupby("Round").pace_exp.rank(method="first")

    preds = {r: fp.predict_quali(r, q, Z, team, circ, dry) for r in rounds_all}
    rows, app, resid_all = [], {}, []
    # pass 1: residual spread of the held-out fit (for the noise term)
    fits = {}
    for r in dry_race:
        tr = base[(base.Round != r) & base.Round.isin(dry_race) & ~base.DNF & base.Position.notna()].dropna(subset=["pace_rank", "qrank"])
        fits[r] = fit_line(tr[["pace_rank", "qrank"]].to_numpy(), tr.Position.to_numpy())
        te = base[(base.Round == r) & ~base.DNF & base.Position.notna()].dropna(subset=["pace_rank"]).copy()
        te["grid_pred"] = te.Driver.map(preds[r].pred_adjusted.rank(method="first")).fillna(te.qrank)
        te = te.dropna(subset=["grid_pred"])
        if len(te) > 5:
            resid_all.append(te.Position.to_numpy() - (fits[r][0] + fits[r][1] * te.pace_rank.to_numpy() + fits[r][2] * te.grid_pred.to_numpy()))
    allres = np.concatenate(resid_all)
    sigma = float(1.4826 * np.median(np.abs(allres - np.median(allres))))
    # retirement rate per driver, shrunk to the field rate
    field_rate = float(base[base.Round.isin(dry_race)].DNF.mean())

    for r in dry_race:
        te = base[base.Round == r].dropna(subset=["pace_rank"]).copy()
        te["grid_pred"] = te.Driver.map(preds[r].pred_adjusted.rank(method="first")).fillna(te.qrank)
        te = te.dropna(subset=["grid_pred"]).reset_index(drop=True)
        if len(te) < 8:
            continue
        b = fits[r]
        mu = b[0] + b[1] * te.pace_rank.to_numpy() + b[2] * te.grid_pred.to_numpy()
        other = base[(base.Round != r) & base.Round.isin(dry_race)]
        pd_ = np.array([(other[other.Driver == dv].DNF.sum() + 3 * field_rate) / (len(other[other.Driver == dv]) + 3) for dv in te.Driver])
        pos, dnf = simulate(mu, pd_, sigma)
        te["exp_pos"] = pos.mean(axis=0)
        te["p_podium"] = (pos <= 3).mean(axis=0)
        te["p_top10"] = (pos <= 10).mean(axis=0)
        te["p_dnf"] = pd_
        app[int(r)] = [dict(driver=x.Driver, team=x.TeamName, exp_pos=float(x.exp_pos), p_podium=float(x.p_podium), p_top10=float(x.p_top10),
                            p_dnf=float(x.p_dnf), actual=(int(x.Position) if np.isfinite(x.Position) and not x.DNF else None))
                       for x in te.sort_values("exp_pos").itertuples()]
        fin = te[~te.DNF & te.Position.notna()]
        if len(fin) < 8:
            continue
        rho = lambda col: float(stats.spearmanr(fin[col], fin.Position).statistic)
        top3_actual = set(fin.nsmallest(3, "Position").Driver)
        row = dict(Round=r, Event=ev.loc[r, "Event"], n=len(fin), rho_model=rho("exp_pos"), rho_grid=rho("grid_pred"), rho_pace=rho("pace_rank"),
                   mae_model=float((fin.exp_pos.rank() - fin.Position.rank()).abs().mean()),
                   mae_grid=float((fin.grid_pred.rank() - fin.Position.rank()).abs().mean()),
                   podium_hits_model=len(set(te.nsmallest(3, "exp_pos").Driver) & top3_actual),
                   podium_hits_grid=len(set(te.nsmallest(3, "grid_pred").Driver) & top3_actual),
                   top10_hits_model=len(set(te.nsmallest(10, "exp_pos").Driver) & set(fin.nsmallest(10, "Position").Driver)))
        rows.append(row)
    B = pd.DataFrame(rows)
    B.to_csv(TAB / "predict_finish_backtest.csv", index=False)
    # calibration of P(podium): bins of predicted vs actual rate
    cal = []
    for r, lst in app.items():
        for x in lst:
            if x["actual"] is not None:
                cal.append((x["p_podium"], int(x["actual"] <= 3)))
    cal = np.array(cal)
    bins = [0, .05, .2, .5, 1.01]
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (cal[:, 0] >= lo) & (cal[:, 0] < hi)
        if m.sum():
            calib.append(dict(lo=lo, hi=min(hi, 1.0), n=int(m.sum()), predicted=float(cal[m, 0].mean()), actual=float(cal[m, 1].mean())))
    S = dict(rounds=int(len(B)), sigma_positions=sigma, field_retire_rate=field_rate,
             rho_model=float(B.rho_model.mean()), rho_grid=float(B.rho_grid.mean()), rho_pace=float(B.rho_pace.mean()),
             mae_model=float(B.mae_model.mean()), mae_grid=float(B.mae_grid.mean()),
             podium_hits_model=float(B.podium_hits_model.mean()), podium_hits_grid=float(B.podium_hits_grid.mean()),
             top10_hits_model=float(B.top10_hits_model.mean()),
             model_better_rho=int((B.rho_model > B.rho_grid).sum()), model_worse_rho=int((B.rho_model < B.rho_grid).sum()),
             podium_calibration=calib)
    json.dump(app, open(OUT / "app_finish.json", "w"), default=float, allow_nan=False)
    json.dump(S, open(OUT / "finish_results.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 200)
    print(json.dumps(S, indent=1, default=float))
    print(B.round(2).to_string())


if __name__ == "__main__":
    run()
