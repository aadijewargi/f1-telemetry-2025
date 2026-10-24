"""Unit and sanity tests for the analysis code and the app data. Run: python tests/test_core.py  (or pytest)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src") if (ROOT / "src").exists() else str(ROOT))
sys.argv = sys.argv[:1]                                   # modules read argv for folders at import
import f1_lines as fl
import f1_strategy as fs
import f1_qrpace as fq
import build_app_data as bad

APP_DATA = next((p for p in (ROOT / "analysis_out" / "app_data.json", ROOT / "results" / "app_data.json") if p.exists()), None)


def circle_lap(R=300.0, n=400, v0=60.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return pd.DataFrame(dict(LapFrac=th / (2 * np.pi), X=10 * R * np.cos(th), Y=10 * R * np.sin(th),
                             Speed=np.full(n, v0 * 3.6), Time_s=th / (2 * np.pi) * (2 * np.pi * R / v0),
                             Brake=np.zeros(n), Throttle=np.full(n, 100.0)))


def test_resample_lap_grid_and_units():
    d = fl.resample_lap(circle_lap(), 200)
    assert len(d["x"]) == 200 and d["u"][0] == 0 and d["u"][-1] < 1
    assert abs(np.hypot(d["x"], d["y"]).mean() - 300) < 1          # X,Y are in 1/10 m
    assert abs(d["v"].mean() - 60) < 0.1                           # km/h -> m/s


def test_curvature_of_circle():
    d = fl.resample_lap(circle_lap(R=300, n=400), 200)
    k, _ = fl.curvature(d["x"], d["y"])
    assert abs(np.median(np.abs(k)) - 1 / 300) < 2e-4
    assert np.all(np.sign(k[5:-5]) == np.sign(k[100]))             # constant turn direction


def test_holm_is_monotone_and_bounded():
    p = np.array([0.001, 0.04, 0.03, 0.5])
    a = fq.holm(p)
    assert np.all(a >= p) and np.all(a <= 1)
    assert np.all(np.diff(a[np.argsort(p)]) >= 0)
    assert abs(a[0] - 0.004) < 1e-12


def make_F(n=60, base=(80.0, 80.5, 81.0), deg=(0.08, 0.04, 0.02)):
    F = np.zeros((3, n + 2, n + 2))
    for c in range(3):
        for age in range(n + 2):
            F[c, age, :] = base[c] + deg[c] * age
    return F


def test_enumerate_plans_rules():
    n, maxage = 50, {0: 28, 1: 35, 2: 44}
    P = fs.enumerate_plans(make_F(n), n, 20.0, maxage)
    assert len(P) > 0 and P.t.is_monotonic_increasing
    for r in P.itertuples():
        assert len(set(r.comps)) >= 2 and 1 <= r.stops <= 3 and len(r.cuts) == r.stops
        b = (0,) + tuple(r.cuts) + (n,)
        lens = [b[i + 1] - b[i] for i in range(r.stops + 1)]
        assert min(lens) >= 8 and all(l <= maxage[c] for l, c in zip(lens, r.comps))


def test_enumerate_plans_prefers_cheap_pit_when_deg_is_low():
    n = 50
    P = fs.enumerate_plans(make_F(n, deg=(0.01, 0.01, 0.01)), n, 25.0, {0: 50, 1: 50, 2: 50})
    assert P.iloc[0].stops == 1                                    # pit loss dominates, so stop once
    P2 = fs.enumerate_plans(make_F(n, deg=(0.6, 0.6, 0.6)), n, 5.0, {0: 50, 1: 50, 2: 50})
    assert P2.iloc[0].stops >= 2                                   # heavy wear and cheap pits, so stop more


def synth_battery_lap(n=250):
    i = np.arange(n)
    v = 60 + 30 * np.sin(2 * np.pi * i / n * 3)                    # m/s
    brk = (np.gradient(v) < -0.35).astype(float)
    thr = np.where(np.gradient(v) > 0, 100.0, 0.0)
    return dict(v=v, brk=brk, thr=thr)


def test_battery_plan_invariants():
    L0 = synth_battery_lap()
    r = bad.battery_for_lap(L0, 20.0, 1.2, lambda v: 12 - 0.02 * v, lambda v: np.full_like(v, -0.02, dtype=float))
    d2, d4 = r["deploy"]["2"], r["deploy"]["4"]
    assert d2["energy_mj"] <= 2.0 + 1e-9 and d4["energy_mj"] <= 4.0 + 1e-9
    assert d4["gain_s"] >= d2["gain_s"] - 1e-9
    assert d2["gain_s"] >= d2["unplanned_gain_s"] - 1e-9           # optimised beats first-come deployment
    ft = (L0["thr"] >= 95) & (L0["brk"] <= 0.5)
    for s in d4["segments"]:
        assert ft[s["a"]:s["b"] + 1].all()                         # deploys only at full throttle
    sel = [z["cap_mj"] for z in r["harvest"]["zones"] if z["selected"]]
    assert sel and sum(sel) >= min(2.0, r["harvest"]["capturable_mj"]) - 1e-9


def test_app_data_is_clean_and_consistent():
    if APP_DATA is None:
        return
    raw = APP_DATA.read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    D = json.loads(raw)
    assert len(D["tracks"]) == 24
    for t in D["tracks"]:
        if t["flags"].get("strategy"):
            s = t["strategy"]
            for p in s["plans"]:
                assert sum(x["laps"] for x in p["stints"]) == s["laps"]
                assert len({x["c"] for x in p["stints"]}) >= 2
            assert min(p["delta_s"] for p in s["plans"]) == 0
        if t["flags"].get("line"):
            ln = t["line"]
            n = ln["n"]
            for d in ln["drivers"].values():
                assert len(d["x"]) == len(d["y"]) == len(d["v"]) == n
            for c in ln["corners"]:
                assert 0 <= c["a"] < n and 0 <= c["b"] < n and 0 <= c["apex"] < n
        if t["flags"].get("predict") and t["predict"].get("winner"):
            assert sum(w["p"] for w in t["predict"]["winner"]) <= 1 + 1e-6
        if t["flags"].get("predict") and t["predict"].get("quali"):
            q = t["predict"]["quali"]
            assert all(0 <= x["p_pole"] <= 1 for x in q) and sum(x["p_pole"] for x in q) <= 1 + 1e-6
            assert [x["gap_s"] for x in q] == sorted(x["gap_s"] for x in q)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
