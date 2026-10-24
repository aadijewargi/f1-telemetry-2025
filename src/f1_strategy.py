#!/usr/bin/env python3
"""
F1 2025: predictive models for (A) tyre strategy and (B) battery harvest/deploy planning.

Usage:  python f1_strategy.py <data_folder> <out_folder>
        (same data folder as f1_analysis.py; run f1_analysis.py first so results/tables exist)

A. Tyre strategy
   1. Lap-time model (gradient boosting, monotone in tyre age and fuel burn) predicting the lap-time delta
      from compound, tyre age, race progress and circuit descriptors. Validated leave-one-round-out.
   2. Pit-loss estimate per circuit from in/out laps.
   3. Strategy optimiser: simulates every legal 1-, 2- and 3-stop plan and picks the fastest.
      Backtested against what race winners actually did.
   4. Pit-window classifier: probability a driver pits within the next 3 laps (leave-one-round-out AUC).

B. Battery plan (physics-based optimiser, NOT a learned model: public telemetry has no state-of-charge channel)
   Rank braking zones for harvesting (120 kW / 2 MJ per lap MGU-K limits) and straights for deployment using
   the marginal lap-time benefit of extra energy, derived from the measured full-throttle acceleration curve.
"""
import sys, json, itertools, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, mean_squared_error, brier_score_loss

warnings.filterwarnings("ignore")
import f1_analysis as fa          # reuses loaders, lap tagging and physics constants

OUT = fa.OUT
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, MUTED, GRID, AXIS = fa.SURF, fa.INK, fa.INK2, fa.MUTED, fa.GRID, fa.AXIS
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CODE = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
NAME = {v: k for k, v in CODE.items()}
S = {}      # results for the report

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


# =============================================================== A1. lap-time model
FEATS = ["comp", "age", "lap_frac", "n_race", "track_c", "air_c", "base_lap", "mean_speed", "top_speed", "brake_kjkg"]


def circuit_table(L, wx):
    d = L[L.Rep & ~L.Wet & L.TelOK]
    c = d.groupby("Round").agg(base_lap=("LapTime_s", "median"), mean_speed=("MeanSpeed_kph", "median"),
                               top_speed=("TopSpeed_kph", "median"), brake_kjkg=("BrakeEnergy_kJkg", "median"),
                               n_race=("N_race", "first"))
    return c.join(wx.set_index("Round")[["TrackTemp_C", "AirTemp_C"]].rename(columns={"TrackTemp_C": "track_c",
                                                                                      "AirTemp_C": "air_c"}))


def lap_table(L, circ):
    d = L[L.Rep & ~L.Wet & L.Compound.isin(CODE) & L.TyreLife.notna() & (L.LapNumber > 1)].copy()
    d["y"] = d.LapTime_s - d.MedLap                      # delta to the driver's own median: removes car and driver speed
    d["comp"] = d.Compound.map(CODE)
    d["age"] = d.TyreLife
    d["lap_frac"] = d.LapNumber / d.N_race
    d = d.join(circ, on="Round", rsuffix="_c")
    d["n_race"] = d["N_race"]
    return d[np.isfinite(d.y) & (d.y.abs() < 8)]


def make_gbm():
    # lap time must not fall with tyre age and must not rise as fuel burns off (lap_frac up -> faster)
    mono = [0 if f not in ("age", "lap_frac") else (1 if f == "age" else -1) for f in FEATS]
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=12, min_samples_leaf=60,
                                         l2_regularization=1.0, monotonic_cst=mono, random_state=0)


def linear_baseline(tr, te):
    """compound-specific age slope + fuel slope, fitted on training rounds only (no circuit knowledge)."""
    X = lambda d: np.column_stack([np.ones(len(d))] + [(d.comp == k) * d.age for k in range(3)] + [d.lap_frac])
    b, *_ = np.linalg.lstsq(X(tr), tr.y.to_numpy(), rcond=None)
    return X(te) @ b


def laptime_model_cv(T):
    rounds = T.Round.to_numpy()
    pred_gbm, pred_lin = np.zeros(len(T)), np.zeros(len(T))
    for tr, te in GroupKFold(n_splits=len(set(rounds))).split(T, groups=rounds):
        m = make_gbm().fit(T.iloc[tr][FEATS], T.iloc[tr].y)
        pred_gbm[te] = m.predict(T.iloc[te][FEATS])
        pred_lin[te] = linear_baseline(T.iloc[tr], T.iloc[te])
    y = T.y.to_numpy()
    rm = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))
    r2 = lambda p: float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    # the useful skill for strategy is the tyre-age trend, so also score slope recovery per round and compound
    rows = []
    for (rnd, comp), g in T.assign(pg=pred_gbm, pl=pred_lin).groupby(["Round", "comp"]):
        if g.age.nunique() > 8 and len(g) > 80:
            s = lambda col: np.polyfit(g.age, g[col], 1)[0]
            rows.append(dict(Round=rnd, comp=comp, obs=s("y"), gbm=s("pg"), lin=s("pl")))
    sl = pd.DataFrame(rows)
    S["laptime_model"] = dict(
        n=int(len(T)), rmse_gbm=rm(pred_gbm), rmse_lin=rm(pred_lin), rmse_zero=rm(np.zeros(len(T))),
        r2_gbm=r2(pred_gbm), r2_lin=r2(pred_lin),
        slope_corr_gbm=float(np.corrcoef(sl.obs, sl.gbm)[0, 1]), slope_corr_lin=float(np.corrcoef(sl.obs, sl.lin)[0, 1]),
        slope_mae_gbm=float((sl.obs - sl.gbm).abs().mean()), slope_mae_lin=float((sl.obs - sl.lin).abs().mean()),
        slope_pairs=int(len(sl)))
    sl.to_csv(TAB / "tyre_slope_cv.csv", index=False)
    return sl


# =============================================================== A2. pit loss
def pit_loss(L):
    d = L[~L.Wet & ~L.Neutral & L.LapTime_s.notna()].sort_values(["Round", "Driver", "LapNumber"])
    rows = []
    for (rnd, drv), g in d.groupby(["Round", "Driver"]):
        g = g.set_index("LapNumber")
        for ln in g.index[g.PitIn.fillna(False).astype(bool)]:
            if ln + 1 in g.index and bool(g.loc[ln + 1, "PitOut"]) and ln > 1:
                rows.append(dict(Round=rnd, Driver=drv, loss=g.loc[ln, "LapTime_s"] + g.loc[ln + 1, "LapTime_s"] - 2 * g.loc[ln, "MedLap"]))
    P = pd.DataFrame(rows)
    P = P[(P.loss > 8) & (P.loss < 45)]
    pl = P.groupby("Round").loss.agg(["median", "count"]).rename(columns={"median": "pit_loss_s"})
    pl.to_csv(TAB / "pit_loss_by_round.csv")
    S["pit_loss"] = dict(median_s=float(P.loss.median()), min_round_s=float(pl.pit_loss_s.min()),
                         max_round_s=float(pl.pit_loss_s.max()), n_stops=int(len(P)))
    return pl.pit_loss_s


# =============================================================== A3. strategy optimiser
def lap_delta_table(model, circ_row, n):
    """F[c, age, lap] = predicted lap-time delta; ages and laps 1..n."""
    ages, laps = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    out = np.full((3, n + 1, n + 1), np.nan)
    for c in range(3):
        X = pd.DataFrame(dict(comp=c, age=ages.ravel(), lap_frac=(laps.ravel() / n), n_race=n,
                              track_c=circ_row.track_c, air_c=circ_row.air_c, base_lap=circ_row.base_lap,
                              mean_speed=circ_row.mean_speed, top_speed=circ_row.top_speed,
                              brake_kjkg=circ_row.brake_kjkg))[FEATS]
        out[c, 1:, 1:] = model.predict(X).reshape(n, n)
    return out


def stint_time(F, c, start, length):
    ages = np.arange(1, length + 1)
    return F[c, ages, start + ages - 1].sum()


def enumerate_plans(F, n, ploss, maxage, min_stint=8, step=2, max_stops=3):
    """Every legal plan: >=2 dry compounds, stints within the observed age envelope. Returns sorted DataFrame."""
    plans = []
    cache = {}

    def st(c, start, length):
        k = (c, start, length)
        if k not in cache:
            cache[k] = stint_time(F, c, start, length)
        return cache[k]

    for stops in range(1, max_stops + 1):
        for comps in itertools.product(range(3), repeat=stops + 1):
            if len(set(comps)) < 2:
                continue
            rng = range(min_stint, n - min_stint + 1, step)
            for cuts in itertools.combinations(rng, stops):
                bounds = (0,) + cuts + (n,)
                lens = [bounds[i + 1] - bounds[i] for i in range(stops + 1)]
                if min(lens) < min_stint or any(l > maxage[c] for l, c in zip(lens, comps)):
                    continue
                t = sum(st(c, bounds[i] + 1, lens[i]) for i, c in enumerate(comps)) + stops * ploss
                plans.append((t, stops, comps, cuts))
    P = pd.DataFrame(plans, columns=["t", "stops", "comps", "cuts"]).sort_values("t").reset_index(drop=True)
    return P


def actual_strategy(L, res, rnd):
    w = res[(res.Round == rnd) & (res.Position == 1)].Abbreviation
    if w.empty:
        return None
    g = L[(L.Round == rnd) & (L.Driver == w.iloc[0])].sort_values("LapNumber")
    st = g.groupby("Stint").agg(comp=("Compound", "first"), start=("LapNumber", "min"), end=("LapNumber", "max"))
    if st.comp.isin(CODE).sum() != len(st):
        return None
    comps = tuple(CODE[c] for c in st.comp)
    cuts = tuple(int(e) for e in st.end.iloc[:-1])
    return w.iloc[0], comps, cuts, int(g.LapNumber.max())


def plan_time(F, n, comps, cuts, ploss):
    bounds = (0,) + tuple(cuts) + (n,)
    return sum(stint_time(F, c, bounds[i] + 1, bounds[i + 1] - bounds[i]) for i, c in enumerate(comps)) + len(cuts) * ploss


def fmt_plan(comps, cuts, n):
    b = (0,) + tuple(cuts) + (n,)
    return " > ".join(f"{NAME[c][0]}{b[i + 1] - b[i]}" for i, c in enumerate(comps))


def strategy_backtest(L, res, T, circ, ploss):
    maxage = {c: float(T[T.comp == c].age.quantile(0.97)) for c in range(3)}
    S["max_stint_laps"] = {NAME[c]: int(maxage[c]) for c in maxage}
    rows, lib = [], {}
    for rnd in sorted(circ.index):
        if rnd not in set(T.Round) or rnd not in ploss.index:
            continue
        tr = T[T.Round != rnd]
        model = make_gbm().fit(tr[FEATS], tr.y)               # held-out circuit: the model has never seen this round
        n = int(circ.loc[rnd, "n_race"])
        F = lap_delta_table(model, circ.loc[rnd], n)
        pl = float(ploss[rnd])
        P = enumerate_plans(F, n, pl, maxage)
        lib[rnd] = (P, F, n, pl)
        act = actual_strategy(L, res, rnd)
        best = P.iloc[0]
        row = dict(Round=rnd, Event=L[L.Round == rnd].Event.iloc[0], laps=n, pit_loss=pl,
                   model_best=fmt_plan(best.comps, best.cuts, n), model_stops=int(best.stops))
        # how much slower is the model's pick than the next best distinct-stop count (robustness of the call)
        alt = P[P.stops != best.stops]
        row["margin_to_other_stop_count_s"] = float(alt.t.iloc[0] - best.t) if len(alt) else np.nan
        if act is not None:
            drv, comps, cuts, nl = act
            row.update(winner=drv, actual=fmt_plan(comps, cuts, n), actual_stops=len(cuts))
            try:
                ta = plan_time(F, n, comps, cuts, pl)
                row["actual_gap_to_model_s"] = float(ta - best.t)
                row["actual_rank_pct"] = float((P.t < ta).mean() * 100)
            except Exception:
                pass
        rows.append(row)
    B = pd.DataFrame(rows)
    B.to_csv(TAB / "strategy_backtest.csv", index=False)
    v = B.dropna(subset=["actual_gap_to_model_s"])
    S["strategy_backtest"] = dict(
        rounds=int(len(B)), rounds_with_winner=int(len(v)),
        stops_match=int((v.model_stops == v.actual_stops).sum()),
        model_one_stop=int((B.model_stops == 1).sum()), model_two_stop=int((B.model_stops == 2).sum()),
        model_three_stop=int((B.model_stops == 3).sum()),
        winner_one_stop=int((v.actual_stops == 1).sum()), winner_two_stop=int((v.actual_stops == 2).sum()),
        winner_three_plus=int((v.actual_stops >= 3).sum()),
        gap_median_s=float(v.actual_gap_to_model_s.median()), gap_mean_s=float(v.actual_gap_to_model_s.mean()),
        within_5s=int((v.actual_gap_to_model_s <= 5).sum()), within_10s=int((v.actual_gap_to_model_s <= 10).sum()),
        winner_top10pct=int((v.actual_rank_pct <= 10).sum()),
        rank_pct_median=float(v.actual_rank_pct.median()))
    return B, lib


def recommend(lib, rnd, lap, comp, age, used=()):
    """
    Live call: best plan from the current state (lap just completed, tyre in use, its age).
    `used` = compound codes already run this race; the two-compound rule needs one more if only one so far.
    Returns (remaining_time_s, [(compound, stop_after_lap), ...]).
    """
    P, F, n, pl = lib[rnd]
    c0 = CODE[comp]
    seen = set(used) | {c0}
    best = None
    for stops in range(0, 3):
        for comps in itertools.product(range(3), repeat=stops):
            if len(seen | set(comps)) < 2:
                continue
            for cuts in itertools.combinations(range(lap + 3, n - 7, 2), stops):
                first_end = cuts[0] if stops else n
                ages = np.arange(age + 1, age + 1 + first_end - lap)
                laps = np.arange(lap + 1, first_end + 1)
                t = F[c0, ages.clip(max=n), laps].sum() + stops * pl
                bounds = (first_end,) + tuple(cuts[1:]) + (n,)
                for i, c in enumerate(comps):
                    t += stint_time(F, c, bounds[i] + 1, bounds[i + 1] - bounds[i])
                if best is None or t < best[0]:
                    best = (float(t), [(NAME[c0], cuts[0] if stops else None)] + [(NAME[c], (cuts[i + 1] if i + 1 < stops else None)) for i, c in enumerate(comps)])
    return best


# =============================================================== A4. pit-window classifier
def pitwindow_model(L, circ):
    d = L[~L.Wet & L.Compound.isin(CODE) & L.TyreLife.notna() & (L.LapNumber > 1)].sort_values(["Round", "Driver", "LapNumber"]).copy()
    d["comp"] = d.Compound.map(CODE)
    d["age"] = d.TyreLife
    d["lap_frac"] = d.LapNumber / d.N_race
    d["laps_left"] = d.N_race - d.LapNumber
    d["pit_soon"] = 0
    for (rnd, drv), g in d.groupby(["Round", "Driver"]):
        pin = g.LapNumber[g.PitIn.fillna(False).astype(bool)].to_numpy()
        ln = g.LapNumber.to_numpy()
        d.loc[g.index, "pit_soon"] = [int(any(0 <= p - l <= 2 for p in pin)) for l in ln]
    # pace trend inside the stint: last lap and last-3 mean vs the stint's best (clean laps only)
    d["clean"] = d.Rep & ~d.Neutral
    d["lapt"] = d.LapTime_s.where(d.clean)
    g = d.groupby(["Round", "Driver", "Stint"])
    d["best_so_far"] = g.lapt.cummin()
    d["pace_loss"] = (d.lapt.groupby([d.Round, d.Driver, d.Stint]).transform(lambda s: s.rolling(3, min_periods=1).mean()) - d.best_so_far)
    d["stint_no"] = d.Stint
    d["sc"] = d.Neutral.astype(int)
    d = d.join(circ[["track_c", "base_lap"]], on="Round")
    d = d[~d.PitOut.fillna(False).astype(bool)]
    feats = ["comp", "age", "lap_frac", "laps_left", "stint_no", "pace_loss", "sc", "track_c"]
    X, y, grp = d[feats], d.pit_soon.to_numpy(), d.Round.to_numpy()
    pr = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=len(set(grp))).split(X, y, grp):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=10, min_samples_leaf=60,
                                           random_state=0).fit(X.iloc[tr], y[tr])
        pr[te] = m.predict_proba(X.iloc[te])[:, 1]
    # baseline: pit probability from tyre age and compound only
    pb = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=len(set(grp))).split(X, y, grp):
        m = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=6, random_state=0).fit(X.iloc[tr][["comp", "age"]], y[tr])
        pb[te] = m.predict_proba(X.iloc[te][["comp", "age"]])[:, 1]
    S["pit_window"] = dict(n=int(len(d)), base_rate=float(y.mean()), auc_full=float(roc_auc_score(y, pr)),
                           auc_age_compound_only=float(roc_auc_score(y, pb)), brier_full=float(brier_score_loss(y, pr)),
                           brier_climatology=float(brier_score_loss(y, np.full(len(y), y.mean()))),
                           auc_by_round_median=float(np.median([roc_auc_score(y[grp == r], pr[grp == r]) for r in set(grp)
                                                                 if 0 < y[grp == r].sum() < (grp == r).sum()])))
    d["p_pit_soon"] = pr
    return d


# =============================================================== B. battery plan
def accel_curve(ftc):
    v = ((ftc.SpeedBin_kph + 5) / 3.6).to_numpy()
    a = ftc.a_g.to_numpy() * fa.G
    k = np.polyfit(v, a, 4)
    return lambda x: np.polyval(k, x), lambda x: np.polyval(np.polyder(k), x)


def battery_plan(tr, wx, ev, ftc, budget_mj=(2.0, 4.0)):
    a_fn, da_fn = accel_curve(ftc)
    m = fa.M_QUALI
    rows, plans = [], {}
    for rnd, g0 in tr[~tr.Round.isin(fa.R["wet_rounds"])].groupby("Round"):
        best = g0.loc[g0.groupby("Driver").LapTime_s.transform("min") == g0.LapTime_s]
        drv = best.groupby("Driver").LapTime_s.first().idxmin()
        g = g0[g0.Driver == drv].sort_values("Distance_m").reset_index(drop=True)
        if len(g) < 100:
            continue
        v = np.maximum(g.Speed.to_numpy() / 3.6, 5.0)
        ds = np.gradient(g.Distance_m.to_numpy())
        dt = ds / v
        ft = (g.Throttle.to_numpy() >= 95) & (g.Brake.to_numpy() == 0)
        brk = g.Brake.to_numpy() == 1
        n = len(g)
        # run ends: the sample where full throttle stops
        run_end = np.zeros(n, int)
        end = n - 1
        for i in range(n - 1, -1, -1):
            if not ft[i]:
                end = i
            run_end[i] = end
        # benefit of one joule at sample i: dv0 = dE/(m v); dv decays as dv' = a'(v)/v dv; dt_saved = -int dv/v^2 ds
        dens = np.zeros(n)
        for i in np.where(ft)[0]:
            j = run_end[i]
            idx = np.arange(i, max(j, i + 1))
            decay = np.exp(np.cumsum(da_fn(v[idx]) / v[idx] * ds[idx]) - da_fn(v[i]) / v[i] * ds[i])
            dens[i] = (1 / (m * v[i])) * np.sum(decay * ds[idx] / v[idx] ** 2)          # s per J
        cap_E = fa.MGUK_KW * 1000 * dt * ft                                               # J deployable per sample at 120 kW
        order = np.argsort(-dens)
        res = dict(Round=rnd, Event=g.Event.iloc[0], driver=drv, lap_s=float(g.LapTime_s.iloc[0]),
                   ft_s=float((dt * ft).sum()), brake_s=float((dt * brk).sum()))
        for B in budget_mj:
            E, left, saved = np.zeros(n), B * 1e6, 0.0
            for i in order:
                if dens[i] <= 0 or left <= 0:
                    break
                e = min(cap_E[i], left)
                E[i], left, saved = e, left - e, saved + e * dens[i]
            # naive: spread the budget over all full-throttle time proportionally
            En = cap_E / cap_E.sum() * min(B * 1e6, cap_E.sum())
            naive = float((En * dens).sum())
            # naive 2: deploy from the start of the lap until the budget is spent (no planning)
            e2, left2 = np.zeros(n), B * 1e6
            for i in range(n):
                if ft[i] and left2 > 0:
                    e2[i] = min(cap_E[i], left2)
                    left2 -= e2[i]
            first = float((e2 * dens).sum())
            res[f"gain_opt_{int(B)}MJ_s"] = saved
            res[f"gain_spread_{int(B)}MJ_s"] = naive
            res[f"gain_first_{int(B)}MJ_s"] = first
            if B == budget_mj[0]:
                dep_idx = np.where(E > 0)[0]
                plans.setdefault(rnd, {})["deploy_zones_m"] = _segments(g.Distance_m.to_numpy(), dep_idx)
        # harvest: contiguous braking zones, capturable energy limited by 120 kW x duration
        zones, i = [], 0
        while i < n:
            if brk[i]:
                j = i
                while j + 1 < n and brk[j + 1]:
                    j += 1
                ke = 0.5 * m * (v[i] ** 2 - v[j] ** 2)
                tz = float(dt[i:j + 1].sum())
                zones.append(dict(start_m=float(g.Distance_m[i]), end_m=float(g.Distance_m[j]), v_in_kph=float(v[i] * 3.6),
                                  v_out_kph=float(v[j] * 3.6), ke_MJ=ke / 1e6, t_s=tz, cap_MJ=min(ke, fa.MGUK_KW * 1000 * tz) / 1e6))
                i = j + 1
            else:
                i += 1
        Z = pd.DataFrame(zones)
        Z = Z[Z.ke_MJ > 0.05].sort_values("cap_MJ", ascending=False).reset_index(drop=True) if len(Z) else Z
        if len(Z):
            Z["cum_MJ"] = Z.cap_MJ.cumsum()
            need = int((Z.cum_MJ < 2.0).sum() + 1) if Z.cum_MJ.iloc[-1] >= 2 else len(Z)
            res.update(zones=len(Z), zones_needed_for_2MJ=need, capturable_total_MJ=float(Z.cap_MJ.sum()),
                       power_share_top_zone=float(Z.cap_MJ.iloc[0] / max(Z.cap_MJ.sum(), 1e-9)),
                       cap_reached=bool(Z.cap_MJ.sum() >= 2.0))
            plans.setdefault(rnd, {})["harvest_zones"] = Z.head(need).round(3).to_dict("records")
        rows.append(res)
    Bt = pd.DataFrame(rows)
    Bt.to_csv(TAB / "battery_plan_by_round.csv", index=False)
    json.dump(plans, open(OUT / "battery_plans.json", "w"), indent=1, default=float)
    S["battery"] = dict(
        rounds=int(len(Bt)),
        gain_opt_2MJ_mean_s=float(Bt.gain_opt_2MJ_s.mean()), gain_spread_2MJ_mean_s=float(Bt.gain_spread_2MJ_s.mean()),
        gain_first_2MJ_mean_s=float(Bt.gain_first_2MJ_s.mean()),
        gain_opt_4MJ_mean_s=float(Bt.gain_opt_4MJ_s.mean()), gain_spread_4MJ_mean_s=float(Bt.gain_spread_4MJ_s.mean()),
        gain_first_4MJ_mean_s=float(Bt.gain_first_4MJ_s.mean()),
        opt_vs_spread_4MJ_mean_s=float((Bt.gain_opt_4MJ_s - Bt.gain_spread_4MJ_s).mean()),
        zones_needed_median=float(Bt.zones_needed_for_2MJ.median()), zones_median=float(Bt.zones.median()),
        cap_reached_rounds=int(Bt.cap_reached.sum()),
        best_event=Bt.loc[Bt.gain_opt_4MJ_s.idxmax(), "Event"], best_gain=float(Bt.gain_opt_4MJ_s.max()),
        worst_event=Bt.loc[Bt.gain_opt_4MJ_s.idxmin(), "Event"], worst_gain=float(Bt.gain_opt_4MJ_s.min()))
    return Bt


def _segments(dist, idx):
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != p + 1:
            out.append((float(dist[s]), float(dist[p])))
            s = i
        p = i
    out.append((float(dist[s]), float(dist[p])))
    return out


# =============================================================== figures
def make_figures(B, sl, Bt, pw, T, lib):
    # F7: tyre slopes observed vs predicted (leave-one-round-out)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    cols = {0: ORANGE, 1: BLUE, 2: AQUA}
    for c in range(3):
        s = sl[sl.comp == c]
        ax.scatter(s.obs, s.gbm, s=48, color=cols[c], edgecolor=SURF, linewidth=2, label=NAME[c].title(), zorder=3)
    lim = [min(sl.obs.min(), sl.gbm.min()) - .01, max(sl.obs.max(), sl.gbm.max()) + .01]
    ax.plot(lim, lim, color=MUTED, lw=1.5)
    ax.set_xlabel("Observed lap-time trend with tyre age (s per lap, includes fuel burn-off)")
    ax.set_ylabel("Predicted trend for a circuit the model never saw")
    ax.set_title("Tyre-trend prediction on held-out circuits")
    ax.legend(frameon=False)
    savefig(fig, "fig7_tyre_model_cv.png")

    # F8: strategy backtest: winner's gap to model optimum
    v = B.dropna(subset=["actual_gap_to_model_s"]).sort_values("actual_gap_to_model_s")
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh(v.Event.str.replace(" Grand Prix", ""), v.actual_gap_to_model_s, color=BLUE, height=.62)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Race winner's actual strategy vs model-optimal (s, simulated, no traffic or safety car)")
    ax.set_title("How far the winners' strategies were from the model's pick")
    ax.grid(axis="y", visible=False)
    savefig(fig, "fig8_strategy_gap.png")

    # F9: example optimal plans vs alternatives for one round
    rnd = int(B.loc[B.model_stops == 2, "Round"].iloc[0]) if (B.model_stops == 2).any() else int(B.Round.iloc[0])
    P, F, n, pl = lib[rnd]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    vals = [(st, P[P.stops == st].t.min() - P.t.min()) for st in (1, 2, 3) if (P.stops == st).any()]
    ax.bar([f"{k} stop" + ("s" if k > 1 else "") for k, _ in vals], [v for _, v in vals], color=BLUE, width=.5)
    for i, (_, v) in enumerate(vals):
        ax.text(i, v + .15, f"+{v:.1f} s", ha="center", fontsize=9, color=INK2)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Best simulated plan vs the overall best (s)")
    ax.set_title(f"{B[B.Round == rnd].Event.iloc[0]}: best plan for each stop count")
    savefig(fig, "fig9_strategy_space.png")

    # F10: pit window probability by tyre age
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for c, col in cols.items():
        g = pw[pw.comp == c]
        t = g.groupby("age").p_pit_soon.mean()
        t = t[g.groupby("age").size() >= 30]
        ax.plot(t.index, t.values, color=col, lw=2, label=NAME[c].title())
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Predicted chance of pitting within 3 laps")
    ax.set_title("Pit-window model: stop probability climbs with tyre age")
    ax.legend(frameon=False)
    savefig(fig, "fig10_pit_window.png")

    # F11: battery: lap time gained from optimal energy placement
    b = Bt.sort_values("gain_opt_4MJ_s")
    fig, ax = plt.subplots(figsize=(8, 5.8))
    y = np.arange(len(b))
    ax.barh(y + .17, b.gain_opt_4MJ_s, height=.3, color=BLUE, label="Planned deployment")
    ax.barh(y - .17, b.gain_first_4MJ_s, height=.3, color=ORANGE, label="Unplanned (spend 4 MJ from the lap start)")
    ax.set_yticks(y)
    ax.set_yticklabels(b.Event.str.replace(" Grand Prix", ""))
    ax.set_xlabel("Estimated lap time saved by a 4 MJ deployment (s)")
    ax.set_title("Where you spend battery energy matters")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2)
    savefig(fig, "fig11_battery_deploy.png")


# =============================================================== main
if __name__ == "__main__":
    d = fa.load()
    L = fa.tag_laps(d["laps"], d["race_control"], d["weather"])
    circ = circuit_table(L, d["weather"])
    T = lap_table(L, circ)
    sl = laptime_model_cv(T)
    ploss = pit_loss(L)
    B, lib = strategy_backtest(L, d["results"], T, circ, ploss)
    pw = pitwindow_model(L, circ)
    ftc = pd.read_csv(TAB / "fullthrottle_curve.csv")
    Bt = battery_plan(d["traces"], d["weather"], d["events"], ftc)
    make_figures(B, sl, Bt, pw, T, lib)
    json.dump(S, open(OUT / "strategy_results.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(json.dumps(S, indent=1, default=float))
    print(B.round(2).to_string())
    print(Bt.round(3).to_string())
