import sys
from pathlib import Path
_R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_R / "src") if (_R / "src").exists() else str(_R))
import numpy as np
import pandas as pd

import f1_season_export as m


def synth_car(hz=4.0, secs=90, cycles=6, noise=0.0, seed=0):
    """Speed oscillates 80..320 km/h; throttle on when accelerating, brake when slowing hard."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, secs, 1 / hz)
    v = 200 + 120 * np.sin(2 * np.pi * cycles * t / secs)          # km/h
    v = v + rng.normal(0, noise, len(t))
    dv = np.gradient(v, t)
    thr = np.where(dv > 0, 100.0, 0.0)
    brk = dv < -20
    return pd.DataFrame({
        "Time": pd.to_timedelta(t, unit="s"),
        "Speed": v, "Throttle": thr, "Brake": brk,
        "nGear": np.clip((v // 45).astype(int), 1, 8),
    })


def test_features_basic():
    f = m.lap_features(synth_car())
    assert abs(f["TopSpeed_kph"] - 320) < 3, f
    assert 0 <= f["Brake_pct"] <= 100 and 0 <= f["FullThrottle_pct"] <= 100
    assert f["FullThrottle_pct"] + f["Brake_pct"] + f["Coast_pct"] <= 100.5, f
    assert f["BrakeZones"] == 6, f["BrakeZones"]
    assert f["DecelP05_g"] < -0.3 and f["AccelP95_g"] > 0.3, f
    assert 190 < f["MeanSpeed_kph"] < 210, f["MeanSpeed_kph"]
    assert abs(f["LapDur_s"] - 89.75) < 0.5


def test_brake_flicker_merged():
    car = synth_car()
    # punch single-sample holes into the brake flag: zones must not multiply
    idx = np.where(car["Brake"].to_numpy())[0]
    car.loc[idx[2::5], "Brake"] = False
    f = m.lap_features(car)
    assert f["BrakeZones"] == 6, f["BrakeZones"]


def test_noise_robust():
    f = m.lap_features(synth_car(noise=1.5))
    assert abs(f["TopSpeed_kph"] - 320) < 8
    assert f["DecelP05_g"] < 0


def test_duplicate_timestamps_and_short():
    car = synth_car()
    car = pd.concat([car, car.iloc[10:20]]).sort_values("Time", kind="stable")
    assert m.lap_features(car) is not None
    assert m.lap_features(synth_car().iloc[:5]) is None
    assert m.lap_features(None) is None


def test_no_braking_lap():
    car = synth_car()
    car["Brake"] = False
    f = m.lap_features(car)
    assert f["BrakeZones"] == 0 and np.isnan(f["BrakeEntrySpeed_kph"])


def test_resample():
    n = 600
    d = np.linspace(0, 5000, n)
    tel = pd.DataFrame({
        "Time": pd.to_timedelta(np.linspace(0, 85, n), unit="s"),
        "Distance": d, "Speed": 200 + 100 * np.sin(d / 400),
        "Throttle": np.full(n, 100.0), "Brake": (np.sin(d / 400) < -0.9),
        "nGear": np.full(n, 6),
        "X": 1000 * np.cos(d / 800), "Y": 1000 * np.sin(d / 800),
    })
    r = m.resample_trace(tel, step=20)
    assert abs(np.diff(r["Distance_m"]).mean() - 20) < 1e-9
    assert len(r) == len(np.arange(0, d[-1], 20))
    assert r["Time_s"].is_monotonic_increasing
    assert set(r["Brake"].unique()) <= {0, 1}
    assert abs(r["Speed"].iloc[0] - tel["Speed"].iloc[0]) < 1e-6
    # non-monotonic distance (stationary samples) must not crash
    tel2 = pd.concat([tel, tel.iloc[100:110]]).sort_values("Time", kind="stable")
    assert m.resample_trace(tel2) is not None


def _expected_cycle():
    """Closed-form values for the synthetic trace v = 200 + 120 sin(2 pi t / 15) km/h."""
    om = 2 * np.pi / 15
    th1 = np.arccos(-20 / (120 * om))              # braking where dv/dt < -20 km/h/s
    th2 = 2 * np.pi - th1
    v1, v2 = (200 + 120 * np.sin(th1)) / 3.6, (200 + 120 * np.sin(th2)) / 3.6
    th_d = np.arcsin(100 / 120)                    # speed above 300 km/h
    return dict(
        shed=6 * 0.5 * (v1 ** 2 - v2 ** 2) / 1000,                       # kJ/kg per lap
        gain=6 * 0.5 * ((320 / 3.6) ** 2 - (80 / 3.6) ** 2) / 1000,
        brake_s=6 * (th2 - th1) / om,
        drs_pct=100 * (np.pi - 2 * th_d) / (2 * np.pi),
    )


def _car_with_rpm_drs():
    car = synth_car()
    car["RPM"] = 6000 + 20 * car["Speed"]
    car["DRS"] = np.where(car["Speed"] > 300, 12, 0)
    return car


def test_lap_physics_closed_form():
    e = _expected_cycle()
    p = m.lap_physics(_car_with_rpm_drs())
    assert abs(p["BrakeEnergy_kJkg"] - e["shed"]) / e["shed"] < 0.10, (p, e)
    assert abs(p["KEGain_kJkg"] - e["gain"]) / e["gain"] < 0.06, (p, e)
    assert abs(p["BrakeTime_s"] - e["brake_s"]) < 2.0, (p, e)
    assert abs(p["DRS_pct"] - e["drs_pct"]) < 2.0, (p, e)
    assert abs(p["RPM_mean"] - 10000) < 150 and abs(p["RPM_max"] - 12400) < 150
    # worked example from the analysis plan: 300 -> 80 km/h at 800 kg is ~2.6 MJ
    ke = 0.5 * 800 * ((300 / 3.6) ** 2 - (80 / 3.6) ** 2)
    assert abs(ke - 2.58e6) < 0.01e6


def test_lap_physics_without_rpm_drs():
    p = m.lap_physics(synth_car())                 # no RPM or DRS columns
    assert np.isnan(p["RPM_mean"]) and p["DRS_pct"] == 0.0
    assert p["BrakeEnergy_kJkg"] > 10
    assert m.lap_physics(synth_car().iloc[:5]) is None


def test_speed_bin_table():
    b = m.speed_bin_table(_car_with_rpm_drs())
    assert set(b["Mode"]) == {"FT", "COAST"}
    ft, co = b[b.Mode == "FT"], b[b.Mode == "COAST"]
    assert 0.5 < ft["sum_a_g"].sum() / ft["n"].sum() < 1.3
    assert -0.5 < co["sum_a_g"].sum() / co["n"].sum() < -0.05
    assert (b.loc[b.DRS == 1, "SpeedBin_kph"] >= 280).all()
    assert (b["sum_a2_g"] >= b["sum_a_g"] ** 2 / b["n"] - 1e-9).all()   # Cauchy-Schwarz
    assert (b["SpeedBin_kph"] % 10 == 0).all()
    # tables from two laps add up
    both = pd.concat([b, b]).groupby(["Mode", "DRS", "SpeedBin_kph"], as_index=False).sum()
    assert both["n"].sum() == 2 * b["n"].sum()
    assert m.speed_bin_table(synth_car().iloc[:5]) is None


def test_resample_carries_rpm_drs():
    n = 600
    d = np.linspace(0, 5000, n)
    tel = pd.DataFrame({
        "Time": pd.to_timedelta(np.linspace(0, 85, n), unit="s"), "Distance": d,
        "Speed": np.full(n, 200.0), "Throttle": np.full(n, 100.0),
        "Brake": np.zeros(n, bool), "nGear": np.full(n, 6), "X": d, "Y": d * 0,
        "RPM": np.linspace(8000, 12000, n), "DRS": np.where(d > 2500, 12, 0)})
    r = m.resample_trace(tel)
    assert abs(r["RPM"].iloc[-1] - 12000) < 40 and set(r["DRS"].unique()) == {0, 12}


def test_parse_rounds():
    allr = list(range(1, 25))
    assert m.parse_rounds("", allr) == allr
    assert m.parse_rounds("1-3,8", allr) == [1, 2, 3, 8]
    assert m.parse_rounds("22-30", allr) == [22, 23, 24]
    assert m.parse_rounds("5", allr) == [5]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
