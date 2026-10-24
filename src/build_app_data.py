#!/usr/bin/env python3
"""
Builds app/data.json for the F1 2025 web app from the analysis outputs.

Usage:  python build_app_data.py <data_folder> <out_folder>
(run f1_analysis.py, f1_strategy.py, f1_lines.py, f1_predict.py first)
"""
import sys, json, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
import f1_analysis as fa
import f1_strategy as fs
import f1_lines as fl

OUT, TAB = fa.OUT, fa.OUT / "tables"
G = fa.G
M = fa.M_QUALI
LET = {0: "S", 1: "M", 2: "H"}


def ints(a, scale=1):
    return [int(round(float(x) * scale)) for x in a]


def battery_for_lap(L0, step, rho, a_fn, da_fn):
    v = np.maximum(L0["v"], 5.0)
    n = len(v)
    ds = np.full(n, step)
    dt = ds / v
    ft = (L0["thr"] >= 95) & (L0["brk"] <= 0.5)
    brk = L0["brk"] > 0.5
    run_end = np.zeros(n, int)
    end = n - 1
    for i in range(n - 1, -1, -1):
        if not ft[i]:
            end = i
        run_end[i] = end
    dens = np.zeros(n)
    for i in np.where(ft)[0]:
        j = run_end[i]
        idx = np.arange(i, max(j, i + 1))
        decay = np.exp(np.cumsum(da_fn(v[idx]) / v[idx] * ds[idx]) - da_fn(v[i]) / v[i] * ds[i])
        dens[i] = (1 / (M * v[i])) * np.sum(decay * ds[idx] / v[idx] ** 2)
    cap = fa.MGUK_KW * 1000 * dt * ft
    order = np.argsort(-dens)
    res = {}
    for B in (2.0, 4.0):
        E, left = np.zeros(n), B * 1e6
        for i in order:
            if dens[i] <= 0 or left <= 0:
                break
            e = min(cap[i], left)
            E[i] = e
            left -= e
        e2, left2 = np.zeros(n), B * 1e6
        for i in range(n):
            if ft[i] and left2 > 0:
                e2[i] = min(cap[i], left2)
                left2 -= e2[i]
        segs, i = [], 0
        while i < n:
            if E[i] > 0:
                j = i
                while j + 1 < n and E[j + 1] > 0:
                    j += 1
                segs.append(dict(a=int(i), b=int(j), kj=float(E[i:j + 1].sum() / 1e3), gain=float((E[i:j + 1] * dens[i:j + 1]).sum()),
                                 v_kph=float(v[i:j + 1].mean() * 3.6)))
                i = j + 1
            else:
                i += 1
        res[str(int(B))] = dict(segments=segs, gain_s=float((E * dens).sum()), unplanned_gain_s=float((e2 * dens).sum()),
                                energy_mj=float(E.sum() / 1e6), ft_capacity_mj=float(cap.sum() / 1e6))
    zones, i = [], 0
    while i < n:
        if brk[i]:
            j = i
            while j + 1 < n and brk[j + 1]:
                j += 1
            ke = 0.5 * M * (v[i] ** 2 - v[j] ** 2)
            tz = float(dt[i:j + 1].sum())
            zones.append(dict(a=int(i), b=int(j), v_in=float(v[i] * 3.6), v_out=float(v[j] * 3.6), ke_mj=float(ke / 1e6), t_s=tz,
                              cap_mj=float(min(ke, fa.MGUK_KW * 1000 * tz) / 1e6)))
            i = j + 1
        else:
            i += 1
    zones = [z for z in zones if z["ke_mj"] > 0.05]
    order_z = sorted(range(len(zones)), key=lambda k: -zones[k]["cap_mj"])
    cum = 0.0
    for rank, k in enumerate(order_z, 1):
        zones[k]["rank"] = rank
        zones[k]["selected"] = bool(cum < 2.0)
        cum += zones[k]["cap_mj"] if cum < 2.0 else 0
    return dict(deploy=res, harvest=dict(zones=zones, capturable_mj=float(sum(z["cap_mj"] for z in zones)),
                                         cap_reached=bool(sum(z["cap_mj"] for z in zones) >= 2.0)))


def main():
    d = fa.load()
    wx = d["weather"]
    wet = set(int(x) for x in wx.loc[wx.RainFraction >= fa.WET_RAIN_FRACTION, "Round"])
    fa.R["wet_rounds"] = sorted(wet)
    L = fa.tag_laps(d["laps"], d["race_control"], wx)
    ev = d["events"].set_index("Round")
    wxi = wx.set_index("Round")
    team_by = d["laps"].drop_duplicates("Driver").set_index("Driver").Team.to_dict()

    # strategy
    circ = fs.circuit_table(L, wx)
    T = fs.lap_table(L, circ)
    ploss = fs.pit_loss(L)
    B, lib = fs.strategy_backtest(L, d["results"], T, circ, ploss)
    Bi = B.set_index("Round")

    # battery curve
    ftc = pd.read_csv(TAB / "fullthrottle_curve.csv")
    a_fn, da_fn = fs.accel_curve(ftc)
    preds = json.load(open(OUT / "app_predictions.json"))
    lines = pd.read_csv(TAB / "lines_corner_driver.csv")

    tracks = []
    tr = d["traces"]
    for rnd in sorted(ev.index):
        info = ev.loc[rnd]
        t = dict(id=int(rnd), name=info.Event.replace(" Grand Prix", " GP"), circuit=info.Circuit, country=info.Country,
                 wet=bool(rnd in wet), flags={})
        # ---- strategy
        if rnd in lib:
            P, F, n, pl = lib[rnd]
            best = P.iloc[0]
            plans = []
            P2 = P.sort_values("t").drop_duplicates(["stops", "comps"])          # best timing for each compound sequence
            chosen = list(P2.head(5).index)
            for stops in (1, 2, 3):                                               # always show the best of each stop count
                k = P2[P2.stops == stops].index
                if len(k) and k[0] not in chosen:
                    chosen.append(k[0])
            for ix in sorted(chosen, key=lambda i: P2.loc[i, "t"]):
                p = P2.loc[ix]
                b_ = (0,) + tuple(p.cuts) + (n,)
                plans.append(dict(stops=int(p.stops), stints=[dict(c=LET[c], laps=int(b_[i + 1] - b_[i])) for i, c in enumerate(p.comps)],
                                  delta_s=float(p.t - best.t)))
            row = Bi.loc[rnd]
            t["strategy"] = dict(laps=int(n), pit_loss_s=float(pl), plans=plans, best_stops=int(best.stops),
                                 margin_other_stops_s=float(row.margin_to_other_stop_count_s) if pd.notna(row.margin_to_other_stop_count_s) else None,
                                 winner=row.get("winner"), actual=row.get("actual"),
                                 actual_gap_s=float(row.actual_gap_to_model_s) if pd.notna(row.get("actual_gap_to_model_s")) else None,
                                 actual_rank_pct=float(row.actual_rank_pct) if pd.notna(row.get("actual_rank_pct")) else None,
                                 track_c=float(wxi.loc[rnd, "TrackTemp_C"]))
            t["flags"]["strategy"] = True
        # ---- traces: map, line, battery
        g0 = tr[(tr.Round == rnd)]
        if len(g0) and rnd not in wet:
            team_of = {k: v for k, v in team_by.items()}
            best_t = g0.groupby("Driver").LapTime_s.first().sort_values()
            drivers = [x for x in best_t.index if g0[g0.Driver == x].LapFrac.nunique() > 100]
            ref = drivers[0]
            lap_len = float(g0[g0.Driver == ref].LapLenEst_m.iloc[0])
            n = int(round(lap_len / 20))
            step = lap_len / n
            laps = {x: fl.resample_lap(g0[g0.Driver == x], n) for x in drivers}
            L0 = laps[ref]
            kap, _ = fl.curvature(L0["x"], L0["y"])
            # corners = prominent speed minima of the fastest lap; each corner spans from the speed peak before to the one after
            from scipy.signal import find_peaks
            v0 = L0["v"] * 3.6
            ext = np.concatenate([v0[-30:], v0, v0[:30]])
            pk, _ = find_peaks(-ext, prominence=10, distance=4)
            apexes = sorted(set(int(p - 30) for p in pk if 30 <= p < 30 + n))
            zones = []
            for ai, ap in enumerate(apexes):
                prev_ap = apexes[ai - 1] if ai else apexes[-1] - n
                next_ap = apexes[ai + 1] if ai + 1 < len(apexes) else apexes[0] + n
                seg = np.arange(prev_ap, ap)
                a = int(seg[np.argmax(v0[seg % n])]) if len(seg) else ap - 3
                seg2 = np.arange(ap, next_ap)
                b_ = int(seg2[np.argmax(v0[seg2 % n])]) if len(seg2) else ap + 3
                a = max(a, ap - 14)
                b_ = min(b_, ap + 14)
                zones.append((a % n if a >= 0 else a + n, b_ % n if b_ < n else b_ - n))
            cx, cy = L0["x"].mean(), L0["y"].mean()
            corners = []
            for k, (a, b) in enumerate(zones, 1):
                rng_ = np.arange(a, b + 1) if b >= a else np.concatenate([np.arange(a, n), np.arange(0, b + 1)])
                ia = int(rng_[np.argmin(L0["v"][rng_])])
                corners.append(dict(k=k, a=int(a), b=int(b), apex=int(ia), dir="L" if kap[ia] > 0 else "R"))
            ld = {}
            for x in drivers:
                ld[x] = dict(team=team_of.get(x, ""), lap_s=float(best_t[x]), x=ints(laps[x]["x"] - cx, 10), y=ints(laps[x]["y"] - cy, 10),
                             v=ints(laps[x]["v"] * 3.6), t=ints(laps[x]["t"], 100))
            t["line"] = dict(n=int(n), step_m=float(step), length_m=lap_len, ref=ref, drivers=ld, corners=corners)
            t["flags"]["line"] = True
            rho = float(wxi.loc[rnd, "AirDensity_kgm3"])
            bat = battery_for_lap(L0, step, rho, a_fn, da_fn)
            bat["ref"] = ref
            bat["brake"] = ints(L0["brk"] > 0.5)
            bat["throttle"] = ints(L0["thr"])
            t["battery"] = bat
            t["flags"]["battery"] = True
        pr = preds.get(str(rnd))
        if pr and (pr.get("has_quali_model") or pr.get("has_race_model")):
            t["predict"] = pr
            t["flags"]["predict"] = True
        tracks.append(t)

    meta = dict(
        generated="2025 season, OpenF1 data",
        results={k: json.load(open(OUT / f"{k}_results.json")) for k in ("strategy", "predict")},
        battery_assumptions=dict(mass_kg=M, mguk_kw=fa.MGUK_KW, deploy_budgets_mj=[2, 4], harvest_cap_mj=2),
        drivers={k: v for k, v in team_by.items()})
    json.dump(dict(meta=meta, tracks=tracks), open(OUT / "app_data.json", "w"), separators=(",", ":"), default=float, allow_nan=False)
    import os
    print("size MB:", round(os.path.getsize(OUT / "app_data.json") / 1e6, 2), "tracks:", len(tracks),
          {k: sum(1 for t in tracks if t["flags"].get(k)) for k in ("strategy", "battery", "line", "predict")})


if __name__ == "__main__":
    main()
