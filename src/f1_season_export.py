#!/usr/bin/env python3
"""
Export a compact summary of a full Formula 1 season using FastF1.

Run this on your own computer (it needs internet access to the F1 timing API).

    pip install fastf1 pandas numpy
    python f1_season_export.py --year 2025 --rounds 1      # quick test, one race
    python f1_season_export.py --year 2025                 # whole season

Outputs (in the folder given by --out, default f1_2025):
    f1_2025_laps.csv            one row per race lap, with pace and telemetry features
    f1_2025_results.csv         race results (grid, finish, status, points)
    f1_2025_quali_traces.csv.gz fastest qualifying lap per driver, resampled every 20 m
                                (speed, throttle, brake, gear, x/y position)

Each round is saved separately in <out>/rounds/, so an interrupted run resumes
where it stopped. Failures are logged to <out>/errors.log and skipped.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

G = 9.80665
FULL_THROTTLE = 99      # % throttle counted as flat out
STEP_M = 20             # distance grid for qualifying traces, metres


# --------------------------------------------------------------------------
# Pure feature code (no FastF1 needed, unit-testable)
# --------------------------------------------------------------------------
def _sec(x):
    return x.total_seconds() if pd.notna(x) else np.nan


def _runs(mask):
    """Index ranges (start, end) of each contiguous True run."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def brake_zones(brk, t, merge_gap=0.6, min_len=0.2):
    """
    Brake zones as (start_idx, end_idx). Dropouts in the brake flag are merged:
    merge_gap is the time between the last braking sample and the next one, so
    0.6 s tolerates one missing sample at ~4 Hz (two sample intervals ~0.5 s).
    """
    merged = []
    for a, b in _runs(brk):
        if merged and t[a] - t[merged[-1][1]] < merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if t[b] - t[a] >= min_len]


def lap_features(car):
    """
    car: DataFrame with Time (timedelta), Speed (km/h), Throttle (%), Brake (bool),
    nGear. Returns a dict of lap-level telemetry features, or None if too short.
    """
    if car is None or len(car) < 10:
        return None
    t = pd.to_timedelta(car["Time"]).dt.total_seconds().to_numpy(float)
    keep = np.r_[True, np.diff(t) > 0]            # drop duplicate/backwards timestamps
    t = t[keep]
    if len(t) < 10:
        return None
    v = car["Speed"].to_numpy(float)[keep] / 3.6   # m/s
    thr = car["Throttle"].to_numpy(float)[keep]
    brk = car["Brake"].to_numpy()[keep].astype(bool)
    gear = car["nGear"].to_numpy(float)[keep] if "nGear" in car else None

    dt = np.diff(t)
    w = np.r_[dt, np.median(dt)]                   # time weight of each sample
    total = w.sum()

    vs = pd.Series(v).rolling(3, center=True, min_periods=1).median().to_numpy()
    acc = np.gradient(vs, t) / G                   # longitudinal accel in g

    zones = brake_zones(brk, t)
    entry = [v[a] * 3.6 for a, _ in zones]
    durs = [t[b] - t[a] for a, b in zones]

    return dict(
        LapDur_s=float(t[-1] - t[0]),
        TopSpeed_kph=float(v.max() * 3.6),
        MeanSpeed_kph=float(np.sum(v * w) / total * 3.6),
        FullThrottle_pct=float(100 * w[thr >= FULL_THROTTLE].sum() / total),
        Brake_pct=float(100 * w[brk].sum() / total),
        Coast_pct=float(100 * w[(thr <= 1) & ~brk].sum() / total),
        BrakeZones=len(zones),
        BrakeEntrySpeed_kph=float(np.mean(entry)) if entry else np.nan,
        BrakeDur_mean_s=float(np.mean(durs)) if durs else np.nan,
        AccelP95_g=float(np.percentile(acc, 95)),
        DecelP05_g=float(np.percentile(acc, 5)),
        Shifts=int(np.sum(np.diff(gear) != 0)) if gear is not None else np.nan,
    )


DRS_OPEN = (10, 12, 14)   # values of the DRS channel that mean the flap is open


def _prep(car):
    """Car-data frame -> dict of aligned arrays with strictly increasing time."""
    if car is None or len(car) < 10:
        return None
    t = pd.to_timedelta(car["Time"]).dt.total_seconds().to_numpy(float)
    keep = np.r_[True, np.diff(t) > 0]
    if keep.sum() < 10:
        return None
    t = t[keep]
    n = len(t)

    def col(name, default):
        return car[name].to_numpy(float)[keep] if name in car else np.full(n, default)

    v = col("Speed", np.nan) / 3.6
    dt = np.diff(t)
    vs = pd.Series(v).rolling(3, center=True, min_periods=1).median().to_numpy()
    return dict(
        t=t, v=v, vs=vs, w=np.r_[dt, np.median(dt)],
        thr=col("Throttle", np.nan),
        brk=car["Brake"].to_numpy()[keep].astype(bool),
        rpm=col("RPM", np.nan),
        drs=np.nan_to_num(col("DRS", 0.0)),
        acc=np.gradient(vs, t) / G,                  # longitudinal accel, g
    )


def lap_physics(car):
    """
    Energy and power features for one lap (per unit mass, so no car mass is
    assumed; multiply by mass later).
      BrakeEnergy_kJkg  kinetic energy shed inside brake zones, kJ per kg
      KEGain_kJkg       total kinetic energy gained over the lap, kJ per kg
                        (inflated slightly by sensor noise)
      BrakeTime_s / CoastTime_s   time braking / off throttle and not braking
      DRS_pct           share of the lap with DRS open (pre-2026 cars only)
      RPM_mean, RPM_max engine speed (time-weighted mean)
    """
    p = _prep(car)
    if p is None:
        return None
    t, vs, brk, w, rpm = p["t"], p["vs"], p["brk"], p["w"], p["rpm"]
    zones = brake_zones(brk, t)
    shed = sum(max(0.0, 0.5 * (vs[a] ** 2 - vs[b] ** 2)) for a, b in zones) / 1000.0
    dke = np.diff(0.5 * vs ** 2)
    ok = np.isfinite(rpm)
    return dict(
        BrakeEnergy_kJkg=float(shed),
        KEGain_kJkg=float(dke[dke > 0].sum() / 1000.0),
        BrakeTime_s=float(w[brk].sum()),
        CoastTime_s=float(w[(p["thr"] <= 1) & ~brk].sum()),
        DRS_pct=float(100 * w[np.isin(p["drs"], DRS_OPEN)].sum() / w.sum()),
        RPM_mean=float(np.sum(rpm[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan,
        RPM_max=float(np.max(rpm[ok])) if ok.any() else np.nan,
    )


def speed_bin_table(car, bin_kph=10):
    """
    Acceleration statistics by speed bin for two driving modes:
      FT     full throttle, not braking  (acceleration curve a(v))
      COAST  off throttle, not braking   (drag + rolling resistance + engine drag)
    Returns sums (n, sum of a, sum of a^2 in g, sum of rpm) so tables from many
    laps can be added together and means/variances recovered later.
    """
    p = _prep(car)
    if p is None:
        return None
    thr, brk = p["thr"], p["brk"]
    ft = (thr >= FULL_THROTTLE) & ~brk
    co = (thr <= 1) & ~brk
    ok = (ft | co) & (np.abs(p["acc"]) < 5) & np.isfinite(p["vs"])
    if not ok.any():
        return None
    df = pd.DataFrame({
        "Mode": np.where(ft, "FT", "COAST")[ok],
        "DRS": np.isin(p["drs"], DRS_OPEN)[ok].astype(int),
        "SpeedBin_kph": (np.floor(p["vs"][ok] * 3.6 / bin_kph) * bin_kph).astype(int),
        "a": p["acc"][ok],
        "a2": p["acc"][ok] ** 2,
        "rpm": np.nan_to_num(p["rpm"])[ok],
    })
    return df.groupby(["Mode", "DRS", "SpeedBin_kph"], as_index=False).agg(
        n=("a", "size"), sum_a_g=("a", "sum"),
        sum_a2_g=("a2", "sum"), sum_rpm=("rpm", "sum"))


def resample_trace(tel, step=STEP_M):
    """Resample a merged telemetry lap onto a fixed distance grid."""
    d = tel["Distance"].to_numpy(float)
    keep = np.r_[True, np.diff(d) > 0]
    d = d[keep]
    if len(d) < 10:
        return None
    grid = np.arange(0.0, d[-1], step)
    t = pd.to_timedelta(tel["Time"]).dt.total_seconds().to_numpy(float)[keep]
    out = {"Distance_m": grid, "Time_s": np.interp(grid, d, t - t[0])}
    for col in ("Speed", "Throttle", "X", "Y"):
        out[col] = np.interp(grid, d, tel[col].to_numpy(float)[keep])
    idx = np.clip(np.searchsorted(d, grid), 0, len(d) - 1)
    out["Brake"] = tel["Brake"].to_numpy()[keep][idx].astype(int)
    out["nGear"] = tel["nGear"].to_numpy()[keep][idx]
    if "RPM" in tel:
        out["RPM"] = np.interp(grid, d, tel["RPM"].to_numpy(float)[keep])
    if "DRS" in tel:
        out["DRS"] = tel["DRS"].to_numpy()[keep][idx]
    return pd.DataFrame(out)


def parse_rounds(spec, all_rounds):
    """'1-5,8' -> [1,2,3,4,5,8] restricted to rounds that exist."""
    if not spec:
        return list(all_rounds)
    wanted = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            wanted.update(range(int(a), int(b) + 1))
        elif part:
            wanted.add(int(part))
    return [r for r in all_rounds if r in wanted]


# --------------------------------------------------------------------------
# FastF1 extraction
# --------------------------------------------------------------------------
def race_tables(ses, rnd, event):
    rows = []
    for _, lap in ses.laps.iterlaps():
        pit_in, pit_out = pd.notna(lap.get("PitInTime")), pd.notna(lap.get("PitOutTime"))
        row = dict(
            Round=rnd, Event=event, Driver=lap["Driver"], Team=lap.get("Team"),
            LapNumber=lap["LapNumber"], Stint=lap.get("Stint"),
            Compound=lap.get("Compound"), TyreLife=lap.get("TyreLife"),
            Position=lap.get("Position"), TrackStatus=lap.get("TrackStatus"),
            IsAccurate=lap.get("IsAccurate"), Deleted=lap.get("Deleted"),
            PitIn=pit_in, PitOut=pit_out,
            LapTime_s=_sec(lap["LapTime"]),
            S1_s=_sec(lap.get("Sector1Time")), S2_s=_sec(lap.get("Sector2Time")),
            S3_s=_sec(lap.get("Sector3Time")),
            SpeedI1=lap.get("SpeedI1"), SpeedI2=lap.get("SpeedI2"),
            SpeedFL=lap.get("SpeedFL"), SpeedST=lap.get("SpeedST"),
        )
        if pd.notna(lap["LapTime"]) and not pit_in and not pit_out:
            try:
                feats = lap_features(lap.get_car_data())
                if feats:
                    row.update(feats)
            except Exception:
                pass  # missing telemetry for this lap: keep the timing row
        rows.append(row)
    laps_df = pd.DataFrame(rows)

    cols = ["Abbreviation", "TeamName", "GridPosition", "Position",
            "ClassifiedPosition", "Status", "Points"]
    res = ses.results.reindex(columns=cols).copy()
    res.insert(0, "Round", rnd)
    res.insert(1, "Event", event)
    return laps_df, res


def quali_traces(ses, rnd, event):
    frames = []
    laps = ses.laps
    pick = getattr(laps, "pick_drivers", None) or laps.pick_driver
    for drv in laps["Driver"].unique():
        try:
            lap = pick(drv).pick_fastest()
            if lap is None or pd.isna(lap["LapTime"]):
                continue
            tr = resample_trace(lap.get_telemetry())
            if tr is None:
                continue
            tr.insert(0, "Driver", drv)
            tr.insert(0, "Event", event)
            tr.insert(0, "Round", rnd)
            tr["LapTime_s"] = _sec(lap["LapTime"])
            frames.append(tr)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(year, out, rounds_spec, do_quali, force):
    import fastf1

    out = Path(out)
    (out / "rounds").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(out / "cache"))
    log = open(out / "errors.log", "a")

    sched = fastf1.get_event_schedule(year, include_testing=False)
    now = pd.Timestamp.now()
    sched = sched[pd.to_datetime(sched["EventDate"]) < now]
    rounds = parse_rounds(rounds_spec, sched["RoundNumber"].tolist())
    print(f"{year}: {len(rounds)} round(s) to process")

    for rnd in rounds:
        event = sched.loc[sched["RoundNumber"] == rnd, "EventName"].iloc[0]
        laps_f = out / "rounds" / f"laps_R{rnd:02d}.csv"
        res_f = out / "rounds" / f"results_R{rnd:02d}.csv"
        tr_f = out / "rounds" / f"traces_R{rnd:02d}.csv.gz"
        t0 = time.time()

        if force or not (laps_f.exists() and res_f.exists()):
            try:
                ses = fastf1.get_session(year, rnd, "R")
                ses.load(laps=True, telemetry=True, weather=False, messages=False)
                laps_df, res_df = race_tables(ses, rnd, event)
                laps_df.to_csv(laps_f, index=False)
                res_df.to_csv(res_f, index=False)
            except Exception:
                log.write(f"R{rnd} {event} race\n{traceback.format_exc()}\n")
                log.flush()
                print(f"  R{rnd:02d} {event}: race FAILED (see errors.log)")

        if do_quali and (force or not tr_f.exists()):
            try:
                ses = fastf1.get_session(year, rnd, "Q")
                ses.load(laps=True, telemetry=True, weather=False, messages=False)
                tr = quali_traces(ses, rnd, event)
                if len(tr):
                    tr.to_csv(tr_f, index=False)
            except Exception:
                log.write(f"R{rnd} {event} quali\n{traceback.format_exc()}\n")
                log.flush()
                print(f"  R{rnd:02d} {event}: qualifying FAILED (see errors.log)")

        print(f"  R{rnd:02d} {event}: done in {time.time() - t0:.0f}s")

    # combine per-round files
    def combine(pattern, name, **kw):
        files = sorted((out / "rounds").glob(pattern))
        if not files:
            return
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        df.to_csv(out / name, index=False, **kw)
        size = (out / name).stat().st_size / 1e6
        print(f"wrote {out / name}  ({len(df):,} rows, {size:.1f} MB)")

    combine("laps_R*.csv", f"f1_{year}_laps.csv")
    combine("results_R*.csv", f"f1_{year}_results.csv")
    combine("traces_R*.csv.gz", f"f1_{year}_quali_traces.csv.gz", compression="gzip")
    log.close()
    print("\nUpload the three f1_*.csv files from", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out", default=None, help="output folder (default f1_<year>)")
    ap.add_argument("--rounds", default="", help="e.g. 1, 1-5, 3,7,9 (default: all)")
    ap.add_argument("--no-quali", action="store_true", help="skip qualifying traces")
    ap.add_argument("--force", action="store_true", help="re-download finished rounds")
    a = ap.parse_args()
    run(a.year, a.out or f"f1_{a.year}", a.rounds, not a.no_quali, a.force)
