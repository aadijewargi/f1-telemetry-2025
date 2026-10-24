import sys
from pathlib import Path
_R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_R / "src") if (_R / "src").exists() else str(_R))
"""Tests for f1_openf1_export using a simulated OpenF1 server (synthetic data)."""
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import f1_openf1_export as m

BASE_T = 1_740_000_000.0          # arbitrary epoch
HZ = 3.7


def iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def t0_of(sk):
    return BASE_T + sk * 10_000


def speed_at(t, t0):
    return 200 + 120 * np.sin(2 * np.pi * (t - t0) / 15.0)      # 6 cycles / 90 s


def make_car(sk, dn, t_start, t_end):
    t0 = t0_of(sk)
    t = np.arange(t_start, t_end, 1 / HZ)
    v = speed_at(t, t0)
    dv = np.gradient(v, t)
    return [dict(date=iso(x), driver_number=dn, session_key=sk, speed=float(s),
                 throttle=100.0 if d > 0 else 0.0, brake=100 if d < -20 else 0,
                 n_gear=int(np.clip(s // 45, 1, 8)), rpm=6000 + 20 * float(s),
                 drs=12 if s > 300 else 0)
            for x, s, d in zip(t, v, dv)]


def make_loc(sk, dn, t_start, t_end):
    t = np.arange(t_start, t_end, 1 / HZ) + 0.05
    ang = 0.07 * t
    return [dict(date=iso(x), driver_number=dn, session_key=sk,
                 x=float(1000 * np.cos(a)), y=float(1000 * np.sin(a)), z=0)
            for x, a in zip(t, ang)]


def m_hz():
    return HZ


def _expected_physics():
    """Closed-form values for v = 200 + 120 sin(2 pi t / 15) km/h over a 90 s lap."""
    om = 2 * np.pi / 15
    th1 = np.arccos(-20 / (120 * om))
    th2 = 2 * np.pi - th1
    v1, v2 = (200 + 120 * np.sin(th1)) / 3.6, (200 + 120 * np.sin(th2)) / 3.6
    th_d = np.arcsin(100 / 120)
    return dict(shed=6 * 0.5 * (v1 ** 2 - v2 ** 2) / 1000,
                gain=6 * 0.5 * ((320 / 3.6) ** 2 - (80 / 3.6) ** 2) / 1000,
                brake_s=6 * (th2 - th1) / om,
                drs_pct=100 * (np.pi - 2 * th_d) / (2 * np.pi))


class FakeApi:
    """Two finished rounds (keys 100/200, 101/201) and one future round."""

    def __init__(self, restrict_car_for=None):
        self.calls = []
        self.restrict_car_for = restrict_car_for

    def get(self, endpoint, **p):
        self.calls.append((endpoint, tuple(sorted(p.items()))))
        sk = p.get("session_key")
        race = sk is not None and sk < 200
        t0 = t0_of(sk) if sk is not None else 0
        if endpoint == "sessions":
            now = datetime.now(timezone.utc).timestamp()
            out = []
            for i, (rk, qk, when) in enumerate([(100, 200, now - 9e6),
                                                (101, 201, now - 5e6),
                                                (102, 202, now + 5e6)], 1):
                out.append(dict(session_key=rk, session_name="Race", meeting_key=i,
                                date_start=iso(when), date_end=iso(when + 7200),
                                is_cancelled=False, location=f"Loc{i}"))
                out.append(dict(session_key=qk, session_name="Qualifying", meeting_key=i,
                                date_start=iso(when - 86400),
                                date_end=iso(when - 86400 + 3600), is_cancelled=False))
            out.append(dict(session_key=1, session_name="Day 1", meeting_key=0,
                            date_start=iso(now - 2e7), date_end=iso(now - 2e7 + 3600)))
            return out
        if endpoint == "meetings":
            return [dict(meeting_key=i, meeting_name=f"GP {i}", circuit_short_name=f"C{i}",
                         circuit_type="Permanent" if i == 1 else "Temporary - Street",
                         country_name=f"Country{i}") for i in (1, 2, 3)]
        if endpoint == "drivers":
            return [dict(driver_number=1, name_acronym="AAA", team_name="TeamX"),
                    dict(driver_number=2, name_acronym="BBB", team_name="TeamY")]
        if endpoint == "laps" and race:
            out = []
            for dn, dur in ((1, 90.0), (2, 91.0)):
                for n in range(1, 6):
                    out.append(dict(driver_number=dn, lap_number=n,
                                    date_start=iso(t0 + (n - 1) * dur), lap_duration=dur,
                                    duration_sector_1=30.0, duration_sector_2=30.0,
                                    duration_sector_3=dur - 60, i1_speed=300, i2_speed=280,
                                    st_speed=330, is_pit_out_lap=(dn == 1 and n == 4)))
            return out
        if endpoint == "laps":                      # qualifying
            out = []
            for dn, durs in ((1, [80.0, 92.0, 93.0]), (2, [82.0, 94.0, 91.0])):
                for n, dur in enumerate(durs, 1):
                    out.append(dict(driver_number=dn, lap_number=n,
                                    date_start=iso(t0 + (n - 1) * 100), lap_duration=dur,
                                    is_pit_out_lap=(n == 1)))
            return out
        if endpoint == "stints":
            return [dict(driver_number=1, stint_number=1, lap_start=1, lap_end=3,
                         compound="MEDIUM", tyre_age_at_start=0),
                    dict(driver_number=1, stint_number=2, lap_start=4, lap_end=None,
                         compound="HARD", tyre_age_at_start=2),
                    dict(driver_number=2, stint_number=1, lap_start=1, lap_end=None,
                         compound="SOFT", tyre_age_at_start=1)]
        if endpoint == "pit":
            return [dict(driver_number=1, lap_number=3, lane_duration=22.0)]
        if endpoint == "car_data":
            if self.restrict_car_for == sk:
                raise m.Restricted("Live F1 session in progress")
            end = t0 + (5 * 91 + 5 if race else 310)
            return make_car(sk, p["driver_number"], t0 - 5, end)
        if endpoint == "location":
            return make_loc(sk, p["driver_number"], t0 - 5, t0 + 310)
        if endpoint == "starting_grid":
            return [dict(driver_number=1, position=2), dict(driver_number=2, position=1)]
        if endpoint == "session_result":
            return [dict(driver_number=1, position=1, dnf=False, dns=False, dsq=False,
                         number_of_laps=5, gap_to_leader=0, duration=460.0),
                    dict(driver_number=2, position=2, dnf=False, dns=False, dsq=False,
                         number_of_laps=5, gap_to_leader=3.2, duration=463.2)]
        if endpoint == "championship_drivers":
            return [dict(driver_number=1, points_current=25, position_current=1),
                    dict(driver_number=2, points_current=18, position_current=2)]
        if endpoint == "race_control":
            return [dict(date=iso(t0 + 100), lap_number=2, category="SafetyCar",
                         flag=None, message="SAFETY CAR DEPLOYED", scope="Track"),
                    dict(date=iso(t0 + 50), lap_number=1, category="Drs",
                         message="DRS ENABLED"),
                    dict(date=iso(t0 + 10), lap_number=1, category="Flag",
                         flag="GREEN", message="GREEN LIGHT", scope="Track")]
        if endpoint == "weather":
            return [dict(date=iso(t0 + 60 * i), air_temperature=25.0,
                         track_temperature=40.0, humidity=50.0, pressure=1013.0,
                         wind_speed=2.0, rainfall=0) for i in range(5)]
        return []


# --------------------------------------------------------------------------
def test_build_events():
    ev = m.build_events(FakeApi(), 2025)
    assert ev["Round"].tolist() == [1, 2]            # future round and testing excluded
    assert ev["RaceKey"].tolist() == [100, 101]
    assert ev["QualiKey"].tolist() == [200, 201]
    assert ev.loc[0, "Event"] == "GP 1" and ev.loc[1, "CircuitType"] == "Temporary - Street"


def test_process_race():
    api = FakeApi()
    ev = m.build_events(api, 2025).iloc[0].to_dict()
    laps, res, rc, bins, wx = m.process_race(api, ev)
    assert len(laps) == 10
    a = laps[laps.Driver == "AAA"].set_index("LapNumber")
    b = laps[laps.Driver == "BBB"].set_index("LapNumber")
    # pit flags, stints, tyre life
    assert bool(a.loc[3, "PitIn"]) and bool(a.loc[4, "PitOut"]) and not bool(a.loc[2, "PitIn"])
    assert a.loc[2, "Compound"] == "MEDIUM" and a.loc[2, "TyreLife"] == 2
    assert a.loc[5, "Compound"] == "HARD" and a.loc[5, "TyreLife"] == 4
    assert b.loc[3, "Compound"] == "SOFT" and b.loc[3, "TyreLife"] == 4
    # telemetry features only on clean laps; pit-in and pit-out laps have none
    assert np.isnan(a.loc[3, "TopSpeed_kph"]) and np.isnan(a.loc[4, "TopSpeed_kph"])
    for n in (1, 2, 5):
        r = a.loc[n]
        assert abs(r["TopSpeed_kph"] - 320) < 4, r["TopSpeed_kph"]
        assert abs(r["LapDur_s"] - 90) < 0.5, r["LapDur_s"]
        assert r["BrakeZones"] == 6, r["BrakeZones"]
        assert 190 < r["MeanSpeed_kph"] < 210
    for n in range(1, 6):
        assert 5 <= b.loc[n, "BrakeZones"] <= 7
        assert abs(b.loc[n, "LapDur_s"] - 91) < 0.5
    assert laps["LapTime_s"].notna().all()
    # results
    assert res.set_index("Abbreviation").loc["AAA", "GridPosition"] == 2
    assert res.set_index("Abbreviation").loc["BBB", "PointsAfter"] == 18
    # race control keeps only SafetyCar and Flag rows
    assert sorted(rc["category"]) == ["Flag", "SafetyCar"]
    # energy / power features on aligned clean laps match closed-form values
    e = _expected_physics()
    for n in (1, 2, 5):
        r = a.loc[n]
        assert abs(r["BrakeEnergy_kJkg"] - e["shed"]) / e["shed"] < 0.10, r["BrakeEnergy_kJkg"]
        assert abs(r["KEGain_kJkg"] - e["gain"]) / e["gain"] < 0.06, r["KEGain_kJkg"]
        assert abs(r["BrakeTime_s"] - e["brake_s"]) < 2.0
        assert abs(r["DRS_pct"] - e["drs_pct"]) < 2.0
        assert abs(r["RPM_mean"] - 10000) < 150
    assert np.isnan(a.loc[3, "BrakeEnergy_kJkg"])
    # speed-bin table: pit laps excluded, phases from lap number, sane accelerations
    assert list(bins.columns) == ["Round", "Event", "Driver", "Phase", "Mode", "DRS",
                                  "SpeedBin_kph", "n", "sum_a_g", "sum_a2_g", "sum_rpm"]
    assert set(bins["Phase"][bins.Driver == "AAA"]) == {0, 2}     # clean laps 1, 2, 5
    assert set(bins["Phase"][bins.Driver == "BBB"]) == {0, 1, 2}
    aaa = bins[bins.Driver == "AAA"]
    samples = 3 * 90 * m_hz()
    assert 0.5 * samples < aaa["n"].sum() < 0.8 * samples, aaa["n"].sum()
    ft = aaa[aaa.Mode == "FT"]
    assert 0.5 < ft["sum_a_g"].sum() / ft["n"].sum() < 1.3
    # weather and air density (ideal gas law: 1013 mbar, 25 C -> 1.1836 kg/m^3)
    assert wx["AirTemp_C"] == 25.0 and wx["Pressure_mbar"] == 1013.0
    assert abs(wx["AirDensity_kgm3"] - 1.1836) < 0.001, wx["AirDensity_kgm3"]
    assert np.isnan(m.air_density(None, 25))


def test_process_quali():
    api = FakeApi()
    ev = m.build_events(api, 2025).iloc[0].to_dict()
    tr = m.process_quali(api, ev)
    assert set(tr["Driver"]) == {"AAA", "BBB"}
    for drv, best in (("AAA", 92.0), ("BBB", 91.0)):      # pit-out lap (shortest) excluded
        d = tr[tr.Driver == drv]
        assert (d["LapTime_s"] == best).all()
        assert d["Distance_m"].is_monotonic_increasing and d["Time_s"].is_monotonic_increasing
        assert abs(np.diff(d["Distance_m"]).mean() - 20) < 1e-9
        assert 0.99 < d["LapFrac"].max() <= 1.0 and d["LapFrac"].min() == 0
        radius = np.hypot(d["X"], d["Y"])
        assert abs(radius - 1000).max() < 5, abs(radius - 1000).max()
        assert 4500 < d["LapLenEst_m"].iloc[0] < 5500
        assert set(d["Brake"].unique()) <= {0, 1}
        assert d["RPM"].between(7500, 12500).all() and set(d["DRS"].unique()) <= {0, 12}


class Resp:
    def __init__(self, code, body=None, text=""):
        self.status_code, self._b, self.text = code, body, text

    def json(self):
        return self._b


class ScriptedHttp:
    def __init__(self, resps):
        self.resps, self.n = list(resps), 0

    def get(self, url, params=None, timeout=None):
        self.n += 1
        r = self.resps.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_http_behaviour():
    m.time.sleep = lambda s: None                      # no waiting in tests
    api = m.Api(delay=0)
    api.http = ScriptedHttp([Resp(429), Resp(503), m.requests.ConnectionError("x"),
                             Resp(200, [{"a": 1}])])
    assert api.get("sessions", year=2025) == [{"a": 1}] and api.http.n == 4
    api.http = ScriptedHttp([Resp(404, text='{"detail":"No results found."}')])
    assert api.get("laps", session_key=1) == []
    api.http = ScriptedHttp([Resp(401, text="Live F1 session in progress")])
    try:
        api.get("sessions", year=2025)
        raise AssertionError("expected Restricted")
    except m.Restricted as e:
        assert "Live F1" in str(e)
    api.http = ScriptedHttp([Resp(500)] * 6)
    try:
        api.get("sessions", year=2025)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_run_end_to_end_and_resume():
    m.time.sleep = lambda s: None
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "o"
        api = FakeApi()
        assert m.run(2025, out, "", True, False, 0, api=api) == 0
        laps = pd.read_csv(out / "f1_2025_laps.csv")
        assert len(laps) == 20 and set(laps["Round"]) == {1, 2}
        assert len(pd.read_csv(out / "f1_2025_results.csv")) == 4
        assert len(pd.read_csv(out / "f1_2025_race_control.csv")) == 4
        tr = pd.read_csv(out / "f1_2025_quali_traces.csv.gz")
        assert set(tr["Round"]) == {1, 2} and set(tr["Driver"]) == {"AAA", "BBB"}
        assert (out / "f1_2025_events.csv").exists()
        sb = pd.read_csv(out / "f1_2025_speedbins.csv.gz")
        assert set(sb["Round"]) == {1, 2} and {"FT", "COAST"} <= set(sb["Mode"])
        wx = pd.read_csv(out / "f1_2025_weather.csv")
        assert len(wx) == 2 and (abs(wx["AirDensity_kgm3"] - 1.1836) < 0.001).all()
        assert "BrakeEnergy_kJkg" in laps.columns and "RPM_mean" in laps.columns
        # resume: nothing re-downloaded except the sessions/meetings listing
        n = len(api.calls)
        assert m.run(2025, out, "", True, False, 0, api=api) == 0
        extra = [c[0] for c in api.calls[n:]]
        assert set(extra) <= {"sessions", "meetings"}, extra
        # a subset of rounds
        assert m.run(2025, Path(tmp) / "o2", "2", False, False, 0, api=FakeApi()) == 0
        assert set(pd.read_csv(Path(tmp) / "o2" / "f1_2025_laps.csv")["Round"]) == {2}
        assert not (Path(tmp) / "o2" / "f1_2025_quali_traces.csv.gz").exists()


def test_restricted_midway_saves_progress():
    m.time.sleep = lambda s: None
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "o"
        code = m.run(2025, out, "", False, False, 0, api=FakeApi(restrict_car_for=101))
        assert code == 2
        laps = pd.read_csv(out / "f1_2025_laps.csv")
        assert set(laps["Round"]) == {1}                # round 1 saved, round 2 not
        assert not (out / "rounds" / "laps_R02.csv").exists()


def test_restricted_at_start():
    class Locked:
        def get(self, *a, **k):
            raise m.Restricted("Live F1 session in progress")
    with tempfile.TemporaryDirectory() as tmp:
        assert m.run(2025, Path(tmp) / "o", "", True, False, 0, api=Locked()) == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
