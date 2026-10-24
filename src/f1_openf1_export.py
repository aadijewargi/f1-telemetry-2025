#!/usr/bin/env python3
"""
Export a compact summary of a full Formula 1 season using the OpenF1 API.

OpenF1 says historical data (2023 onwards) is free without authentication.
While a live F1 session is running the server may refuse requests (HTTP 401);
this script then stops cleanly. Run the same command again after the session
ends and it resumes where it stopped.

Keep this file in the SAME folder as f1_season_export.py (it reuses the
tested calculation code from there).

    pip install requests pandas numpy
    python f1_openf1_export.py --year 2025 --rounds 1      # quick test, one race
    python f1_openf1_export.py --year 2025                 # whole season

Outputs (in --out, default f1_<year>):
    f1_<year>_events.csv            one row per round: event, circuit, circuit type
    f1_<year>_laps.csv              one row per race lap: timing, tyres, telemetry features
    f1_<year>_results.csv           race results: grid, finish, DNF flags, points after race
    f1_<year>_race_control.csv      safety car and flag messages (for filtering neutralised laps)
    f1_<year>_quali_traces.csv.gz   fastest qualifying lap per driver every 20 m
                                    (speed, throttle, brake, gear, rpm, DRS, x/y position)
    f1_<year>_speedbins.csv.gz      acceleration vs speed at full throttle and while
                                    coasting, per driver, race phase and DRS state
                                    (for drag, power and fuel-mass estimates)
    f1_<year>_weather.csv           race-session air/track temperature, pressure,
                                    humidity, wind and the air density they imply
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from f1_season_export import (lap_features, lap_physics, parse_rounds,
                              resample_trace, speed_bin_table)

BASE = "https://api.openf1.org/v1"


class Restricted(Exception):
    """The server answered 401 (e.g. live session in progress)."""


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Api:
    def __init__(self, delay=0.5):
        self.delay = delay
        self.http = requests.Session()
        self.http.headers["User-Agent"] = "f1-season-export (personal analysis script)"

    def get(self, endpoint, **params):
        """GET a list of records. Retries on rate limits and server errors."""
        url = f"{BASE}/{endpoint}"
        last = ""
        for attempt in range(6):
            time.sleep(self.delay)
            try:
                r = self.http.get(url, params=params, timeout=90)
            except requests.RequestException as e:
                last = repr(e)
                time.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, list) else []
            if r.status_code == 404:        # OpenF1 uses 404 for "no results"
                return []
            if r.status_code == 401:
                raise Restricted(r.text[:300])
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(min(60, 3 * 2 ** attempt))
                continue
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            break
        raise RuntimeError(f"{endpoint} {params}: {last}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ts(x):
    """ISO string -> epoch seconds (naive strings are treated as UTC)."""
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.timestamp()


def epoch_s(series):
    # format="ISO8601" accepts mixed forms (with/without fractional seconds)
    s = pd.to_datetime(series, utc=True, format="ISO8601")
    return (s - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(float)


def car_frame(rows):
    """car_data records -> DataFrame sorted by time (column t = epoch seconds)."""
    if not rows:
        return None
    d = pd.DataFrame(rows)
    need = ["date", "speed", "throttle", "brake", "n_gear"]
    if any(c not in d for c in need):
        return None
    d = d.dropna(subset=need)
    if d.empty:
        return None
    d = d.assign(t=epoch_s(d["date"])).sort_values("t").drop_duplicates("t")
    return d.reset_index(drop=True)


def lapframe(seg):
    """Slice of car data -> the frame layout lap_features() expects."""
    if seg is None or len(seg) < 10:
        return None
    t = seg["t"].to_numpy(float)
    return pd.DataFrame({
        "Time": pd.to_timedelta(t - t[0], unit="s"),
        "Speed": seg["speed"].to_numpy(float),
        "Throttle": seg["throttle"].to_numpy(float),
        "Brake": seg["brake"].to_numpy(float) > 0,
        "nGear": seg["n_gear"].to_numpy(float),
        "RPM": seg["rpm"].to_numpy(float) if "rpm" in seg else np.nan,
        "DRS": seg["drs"].to_numpy(float) if "drs" in seg else 0.0,
    })


def fetch_car(api, sk, dn):
    try:
        return car_frame(api.get("car_data", session_key=sk, driver_number=dn))
    except Restricted:
        raise
    except Exception as e:
        print(f"    warning: car_data failed for driver {dn}: {str(e)[:100]}")
        return None


def stint_info(stints, lap):
    """(stint number, compound, tyre age in laps incl. this one) for a lap."""
    for s in stints:
        ls, le = s.get("lap_start"), s.get("lap_end")
        if ls is not None and lap >= ls and (le is None or lap <= le):
            age = (s.get("tyre_age_at_start") or 0) + (lap - ls) + 1
            return s.get("stint_number"), s.get("compound"), age
    return None, None, np.nan


def build_tel(car, loc, t0, t1):
    """
    Merge car data and position over one lap into the layout resample_trace()
    expects. Distance is integrated from speed (OpenF1 has no distance channel),
    so it is approximate: use LapFrac when comparing drivers.
    """
    if car is None or not loc:
        return None
    t = car["t"].to_numpy(float)
    i0, i1 = np.searchsorted(t, t0, "left"), np.searchsorted(t, t1, "right")
    seg = car.iloc[i0:i1]
    if len(seg) < 10:
        return None
    ld = pd.DataFrame(loc).dropna(subset=["date", "x", "y"])
    if len(ld) < 10:
        return None
    ld = ld.assign(t=epoch_s(ld["date"])).sort_values("t").drop_duplicates("t")
    st = seg["t"].to_numpy(float)
    v = seg["speed"].to_numpy(float) / 3.6
    dist = np.r_[0.0, np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(st))]
    return pd.DataFrame({
        "Time": pd.to_timedelta(st - st[0], unit="s"),
        "Distance": dist,
        "Speed": seg["speed"].to_numpy(float),
        "Throttle": seg["throttle"].to_numpy(float),
        "Brake": seg["brake"].to_numpy(float) > 0,
        "nGear": seg["n_gear"].to_numpy(float),
        "X": np.interp(st, ld["t"].to_numpy(float), ld["x"].to_numpy(float)),
        "Y": np.interp(st, ld["t"].to_numpy(float), ld["y"].to_numpy(float)),
        "RPM": seg["rpm"].to_numpy(float) if "rpm" in seg else np.nan,
        "DRS": seg["drs"].to_numpy(float) if "drs" in seg else 0.0,
    })


# --------------------------------------------------------------------------
# Season structure
# --------------------------------------------------------------------------
def build_events(api, year):
    sessions = api.get("sessions", year=year)
    meetings = {m["meeting_key"]: m for m in api.get("meetings", year=year)}
    races = [s for s in sessions
             if s.get("session_name") == "Race" and not s.get("is_cancelled")]
    races.sort(key=lambda s: s["date_start"])
    quali = {s["meeting_key"]: s for s in sessions
             if s.get("session_name") == "Qualifying" and not s.get("is_cancelled")}
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for rnd, s in enumerate(races, 1):
        end = pd.Timestamp(s.get("date_end") or s["date_start"])
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        if end > now:
            continue                                   # not finished yet
        mk = s["meeting_key"]
        m = meetings.get(mk, {})
        rows.append(dict(
            Round=rnd, MeetingKey=mk,
            Event=m.get("meeting_name") or s.get("location"),
            Circuit=m.get("circuit_short_name") or s.get("circuit_short_name"),
            CircuitType=m.get("circuit_type"), Country=m.get("country_name"),
            Date=s["date_start"], RaceKey=s["session_key"],
            QualiKey=quali.get(mk, {}).get("session_key"),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Per-round extraction
# --------------------------------------------------------------------------
def air_density(p_mbar, t_c):
    """Dry-air density from the ideal gas law, kg/m^3 (humid air is ~1% lighter)."""
    try:
        p, t = float(p_mbar), float(t_c)
    except (TypeError, ValueError):
        return np.nan
    if not (np.isfinite(p) and np.isfinite(t)):
        return np.nan
    return p * 100.0 / (287.05 * (t + 273.15))


def weather_row(api, ev):
    """Race-session mean weather plus the air density it implies."""
    w = pd.DataFrame(api.get("weather", session_key=ev["RaceKey"]))
    row = dict(Round=ev["Round"], Event=ev["Event"])
    for src, dst in (("air_temperature", "AirTemp_C"), ("track_temperature", "TrackTemp_C"),
                     ("humidity", "Humidity_pct"), ("pressure", "Pressure_mbar"),
                     ("wind_speed", "WindSpeed_ms"), ("rainfall", "RainFraction")):
        row[dst] = (float(pd.to_numeric(w[src], errors="coerce").mean())
                    if src in w else np.nan)
    row["AirDensity_kgm3"] = air_density(row["Pressure_mbar"], row["AirTemp_C"])
    return row


def driver_bins(drv_bins, driver, rnd, event, total_laps, slow_factor=1.07):
    """
    Add up per-lap speed-bin tables over representative laps only (lap time
    within 107% of the driver's median, which drops safety-car and damaged
    laps), split into three race phases (early/mid/late, i.e. fuel load).
    """
    if not drv_bins:
        return None
    med = float(np.median([d for _, d, _ in drv_bins]))
    parts = []
    for n, dur, b in drv_bins:
        if dur <= slow_factor * med:
            b = b.copy()
            b["Phase"] = min(2, int(3 * (n - 1) / max(total_laps, 1)))
            parts.append(b)
    if not parts:
        return None
    out = (pd.concat(parts)
           .groupby(["Phase", "Mode", "DRS", "SpeedBin_kph"], as_index=False)
           [["n", "sum_a_g", "sum_a2_g", "sum_rpm"]].sum())
    out.insert(0, "Driver", driver)
    out.insert(0, "Event", event)
    out.insert(0, "Round", rnd)
    return out


def process_race(api, ev):
    rnd, event, sk = ev["Round"], ev["Event"], ev["RaceKey"]
    drivers = {d["driver_number"]: d for d in api.get("drivers", session_key=sk)}
    laps = api.get("laps", session_key=sk)
    stints = {}
    for s in api.get("stints", session_key=sk):
        stints.setdefault(s["driver_number"], []).append(s)
    pits = {(p["driver_number"], p["lap_number"])
            for p in api.get("pit", session_key=sk)}
    by_driver = {}
    for lp in laps:
        by_driver.setdefault(lp["driver_number"], []).append(lp)

    total_laps = max((lp["lap_number"] for lp in laps), default=1)
    bin_frames = []
    rows = []
    for dn in sorted(by_driver):
        d = drivers.get(dn, {})
        car = fetch_car(api, sk, dn)
        ct = car["t"].to_numpy(float) if car is not None else None
        drv_bins = []
        for lp in sorted(by_driver[dn], key=lambda x: x["lap_number"]):
            n = lp["lap_number"]
            stint, comp, age = stint_info(stints.get(dn, []), n)
            pit_in = (dn, n) in pits
            pit_out = bool(lp.get("is_pit_out_lap"))
            dur, start = lp.get("lap_duration"), lp.get("date_start")
            row = dict(
                Round=rnd, Event=event,
                Driver=d.get("name_acronym", str(dn)), DriverNumber=dn,
                Team=d.get("team_name"), LapNumber=n, Stint=stint,
                Compound=comp, TyreLife=age, PitIn=pit_in, PitOut=pit_out,
                LapStart=start,
                LapTime_s=dur if dur is not None else np.nan,
                S1_s=lp.get("duration_sector_1"), S2_s=lp.get("duration_sector_2"),
                S3_s=lp.get("duration_sector_3"),
                SpeedI1=lp.get("i1_speed"), SpeedI2=lp.get("i2_speed"),
                SpeedST=lp.get("st_speed"),
            )
            if car is not None and dur and start and not pit_in and not pit_out:
                t0 = ts(start)
                i0 = np.searchsorted(ct, t0, "left")
                i1 = np.searchsorted(ct, t0 + dur, "right")
                frame = lapframe(car.iloc[i0:i1])
                feats = lap_features(frame)
                if feats:
                    row.update(feats)
                    phys = lap_physics(frame)
                    if phys:
                        row.update(phys)
                    b = speed_bin_table(frame)
                    if b is not None:
                        drv_bins.append((n, dur, b))
            rows.append(row)
        bin_frames.append(driver_bins(drv_bins, d.get("name_acronym", str(dn)),
                                      rnd, event, total_laps))
    laps_df = pd.DataFrame(rows)
    bin_frames = [f for f in bin_frames if f is not None]
    bins_df = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()

    grid = {g["driver_number"]: g.get("position")
            for g in api.get("starting_grid", session_key=sk)}
    champ = {}
    try:
        for c in api.get("championship_drivers", session_key=sk):
            champ[c["driver_number"]] = c
    except Restricted:
        raise
    except Exception:
        pass
    res_rows = []
    for r in api.get("session_result", session_key=sk):
        dn = r["driver_number"]
        d = drivers.get(dn, {})
        res_rows.append(dict(
            Round=rnd, Event=event, DriverNumber=dn,
            Abbreviation=d.get("name_acronym"), TeamName=d.get("team_name"),
            GridPosition=grid.get(dn), Position=r.get("position"),
            DNF=r.get("dnf"), DNS=r.get("dns"), DSQ=r.get("dsq"),
            Laps=r.get("number_of_laps"), GapToLeader=r.get("gap_to_leader"),
            Duration_s=r.get("duration"),
            PointsAfter=champ.get(dn, {}).get("points_current"),
            ChampPosAfter=champ.get(dn, {}).get("position_current"),
        ))
    res_df = pd.DataFrame(res_rows)

    rc_cols = ["date", "lap_number", "category", "flag", "message",
               "scope", "sector", "driver_number"]
    rc = [r for r in api.get("race_control", session_key=sk)
          if r.get("category") in ("SafetyCar", "Flag")]
    rc_df = pd.DataFrame(rc).reindex(columns=rc_cols)
    rc_df.insert(0, "Round", rnd)
    rc_df.insert(1, "Event", event)
    wx = weather_row(api, ev)
    return laps_df, res_df, rc_df, bins_df, wx


def process_quali(api, ev):
    rnd, event, sk = ev["Round"], ev["Event"], ev["QualiKey"]
    drivers = {d["driver_number"]: d for d in api.get("drivers", session_key=sk)}
    best = {}
    for lp in api.get("laps", session_key=sk):
        dur = lp.get("lap_duration")
        if not dur or lp.get("is_pit_out_lap") or not lp.get("date_start"):
            continue
        dn = lp["driver_number"]
        if dn not in best or dur < best[dn]["lap_duration"]:
            best[dn] = lp
    frames = []
    for dn, lp in sorted(best.items()):
        try:
            car = fetch_car(api, sk, dn)
            loc = api.get("location", session_key=sk, driver_number=dn)
            t0 = ts(lp["date_start"])
            tel = build_tel(car, loc, t0, t0 + lp["lap_duration"])
            if tel is None:
                continue
            tr = resample_trace(tel)
            if tr is None:
                continue
            total = float(tel["Distance"].iloc[-1])
            tr["LapFrac"] = tr["Distance_m"] / total if total > 0 else np.nan
            tr.insert(0, "Driver", drivers.get(dn, {}).get("name_acronym", str(dn)))
            tr.insert(0, "Event", event)
            tr.insert(0, "Round", rnd)
            tr["LapTime_s"] = lp["lap_duration"]
            tr["LapLenEst_m"] = total
            frames.append(tr)
        except Restricted:
            raise
        except Exception as e:
            print(f"    warning: quali trace failed for driver {dn}: {str(e)[:100]}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(year, out, rounds_spec, do_quali, force, delay, api=None):
    api = api or Api(delay)
    out = Path(out)
    (out / "rounds").mkdir(parents=True, exist_ok=True)
    log = open(out / "errors.log", "a")
    code = 0

    try:
        events = build_events(api, year)
    except Restricted as e:
        print("\nOpenF1 is refusing requests right now (HTTP 401).")
        print("Server says:", e)
        print("Usually this means a live F1 session is in progress. "
              "Try again after it ends.")
        return 2
    if events.empty:
        print(f"No finished races found for {year}.")
        return 1
    events.to_csv(out / f"f1_{year}_events.csv", index=False)

    rounds = parse_rounds(rounds_spec, events["Round"].tolist())
    print(f"{year}: {len(rounds)} round(s) to process")

    for rnd in rounds:
        ev = events[events["Round"] == rnd].iloc[0].to_dict()
        event = ev["Event"]
        laps_f = out / "rounds" / f"laps_R{rnd:02d}.csv"
        res_f = out / "rounds" / f"results_R{rnd:02d}.csv"
        rc_f = out / "rounds" / f"race_control_R{rnd:02d}.csv"
        tr_f = out / "rounds" / f"traces_R{rnd:02d}.csv.gz"
        bins_f = out / "rounds" / f"speedbins_R{rnd:02d}.csv.gz"
        wx_f = out / "rounds" / f"weather_R{rnd:02d}.csv"
        t0 = time.time()
        try:
            if force or not all(f.exists() for f in (laps_f, res_f, rc_f, bins_f, wx_f)):
                laps_df, res_df, rc_df, bins_df, wx = process_race(api, ev)
                bins_df.to_csv(bins_f, index=False, compression="gzip")
                pd.DataFrame([wx]).to_csv(wx_f, index=False)
                laps_df.to_csv(laps_f, index=False)
                res_df.to_csv(res_f, index=False)
                rc_df.to_csv(rc_f, index=False)
        except Restricted as e:
            print(f"\n  R{rnd:02d} {event}: OpenF1 returned 401: {e}")
            print("  Stopping. Finished rounds are saved; rerun later to continue.")
            code = 2
            break
        except Exception:
            log.write(f"R{rnd} {event} race\n{traceback.format_exc()}\n")
            log.flush()
            print(f"  R{rnd:02d} {event}: race FAILED (see errors.log)")

        qk = ev.get("QualiKey")
        if do_quali and pd.notna(qk) and (force or not tr_f.exists()):
            try:
                tr = process_quali(api, ev)
                if len(tr):
                    tr.to_csv(tr_f, index=False)
            except Restricted as e:
                print(f"\n  R{rnd:02d} {event}: OpenF1 returned 401: {e}")
                print("  Stopping. Finished rounds are saved; rerun later to continue.")
                code = 2
                break
            except Exception:
                log.write(f"R{rnd} {event} quali\n{traceback.format_exc()}\n")
                log.flush()
                print(f"  R{rnd:02d} {event}: qualifying FAILED (see errors.log)")

        print(f"  R{rnd:02d} {event}: done in {time.time() - t0:.0f}s")

    def combine(pattern, name, **kw):
        files = sorted((out / "rounds").glob(pattern))
        if not files:
            return
        frames = []
        for f in files:
            try:
                frames.append(pd.read_csv(f))
            except pd.errors.EmptyDataError:
                pass                                   # a round with no rows
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out / name, index=False, **kw)
        size = (out / name).stat().st_size / 1e6
        print(f"wrote {out / name}  ({len(df):,} rows, {size:.1f} MB)")

    combine("laps_R*.csv", f"f1_{year}_laps.csv")
    combine("results_R*.csv", f"f1_{year}_results.csv")
    combine("race_control_R*.csv", f"f1_{year}_race_control.csv")
    combine("traces_R*.csv.gz", f"f1_{year}_quali_traces.csv.gz", compression="gzip")
    combine("speedbins_R*.csv.gz", f"f1_{year}_speedbins.csv.gz", compression="gzip")
    combine("weather_R*.csv", f"f1_{year}_weather.csv")
    log.close()

    produced = sorted(p.name for p in out.glob(f"f1_{year}_*"))
    if produced:
        print("\nUpload these files from", out)
        for name in produced:
            print("  ", name)
    else:
        print("\nNothing was written. Check errors.log in", out)
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out", default=None, help="output folder (default f1_<year>)")
    ap.add_argument("--rounds", default="", help="e.g. 1, 1-5, 3,7,9 (default: all)")
    ap.add_argument("--no-quali", action="store_true", help="skip qualifying traces")
    ap.add_argument("--force", action="store_true", help="re-download finished rounds")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="seconds to wait between requests (default 0.5)")
    a = ap.parse_args()
    sys.exit(run(a.year, a.out or f"f1_{a.year}", a.rounds,
                 not a.no_quali, a.force, a.delay))
