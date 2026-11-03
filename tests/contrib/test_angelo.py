from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.contrib.angelo import bar_log_return, bar_vwap, close_location, close_location_residual
from src.stats import information_coefficient

TRAIN = Path("data/train_1h.parquet")


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    if not TRAIN.exists():
        pytest.skip("training panel not available")
    return pd.read_parquet(TRAIN)


def _bars(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("open", "high", "low"):
        if col not in df:
            df[col] = df["close"]
    return df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


# --- bar_vwap -------------------------------------------------------------------------------


def test_vwap_inside_bar_range_on_train(train: pd.DataFrame) -> None:
    vwap = bar_vwap(train)
    assert vwap.notna().all()
    # Allow only float rounding on the boundary.
    tol = 1e-12 * train["high"].abs()
    assert (vwap >= train["low"] - tol).all()
    assert (vwap <= train["high"] + tol).all()


def test_vwap_hand_built_bars_and_zero_volume() -> None:
    df = _bars(
        [
            {"timestamp": "2025-01-01 00:00", "symbol": "A", "close": 101.0, "volume": 10.0, "quote_volume": 1000.0},
            {"timestamp": "2025-01-01 00:00", "symbol": "B", "close": 50.0, "volume": 0.0, "quote_volume": 0.0},
            {"timestamp": "2025-01-01 00:00", "symbol": "C", "close": 2.0, "volume": 4.0, "quote_volume": 10.0},
        ]
    )
    vwap = bar_vwap(df)
    assert vwap.iloc[0] == pytest.approx(100.0)
    assert np.isnan(vwap.iloc[1])
    assert vwap.iloc[2] == pytest.approx(2.5)

    cl = close_location(df)
    assert cl.iloc[0] == pytest.approx(np.log(101.0 / 100.0))
    assert np.isnan(cl.iloc[1])
    assert cl.iloc[2] == pytest.approx(np.log(2.0 / 2.5))
    assert list(vwap.index) == list(df.index)


# --- close_location_residual ----------------------------------------------------------------


def test_residual_hand_built_cross_section() -> None:
    # Two hours, four symbols. Hour 2: close_location = 0.5 * return + 0.01 except D, which is off the line.
    prev = [100.0, 100.0, 100.0, 100.0]
    rets = [0.02, -0.01, 0.00, 0.01]
    extra = [0.0, 0.0, 0.0, 0.004]
    rows = []
    for sym, p in zip("ABCD", prev):
        rows.append({"timestamp": "2025-01-01 00:00", "symbol": sym, "close": p, "volume": 1.0, "quote_volume": p})
    for sym, p, r, e in zip("ABCD", prev, rets, extra):
        c = p * np.exp(r)
        cl = 0.5 * r + 0.01 + e
        vwap = c / np.exp(cl)
        rows.append({"timestamp": "2025-01-01 01:00", "symbol": sym, "close": c, "volume": 2.0, "quote_volume": 2.0 * vwap})
    df = _bars(rows)

    res = close_location_residual(df)
    assert res.iloc[:4].isna().all()  # first row of every symbol has no return

    h2 = res.iloc[4:].to_numpy()
    r = np.array(rets)
    e = np.array(extra)
    # Expected: residual of e on [1, r] (the exact line drops out).
    X = np.column_stack([np.ones(4), r])
    beta, *_ = np.linalg.lstsq(X, e, rcond=None)
    np.testing.assert_allclose(h2, e - X @ beta, atol=1e-12)


def test_residual_constant_return_cross_section() -> None:
    rows = []
    for sym, cl in zip("ABC", [0.001, 0.002, 0.006]):
        rows.append({"timestamp": "2025-01-01 00:00", "symbol": sym, "close": 10.0, "volume": 1.0, "quote_volume": 10.0})
        rows.append({"timestamp": "2025-01-01 01:00", "symbol": sym, "close": 11.0, "volume": 1.0, "quote_volume": 11.0 / np.exp(cl)})
    df = _bars(rows)
    res = close_location_residual(df).iloc[3:].to_numpy()
    np.testing.assert_allclose(res, np.array([0.001, 0.002, 0.006]) - 0.003, atol=1e-15)


def test_residual_orthogonal_to_return_on_train(train: pd.DataFrame) -> None:
    res = close_location_residual(train)
    r = bar_log_return(train)
    assert res[r.isna()].isna().all()
    assert res[r.notna()].notna().all()

    ok = res.notna()
    g = pd.DataFrame({"ts": train["timestamp"][ok], "e": res[ok], "r": r[ok]})
    g["rd"] = g["r"] - g.groupby("ts")["r"].transform("mean")
    g["er"] = g["e"] * g["rd"]
    per_ts = g.groupby("ts").agg(mean_e=("e", "mean"), cov=("er", "mean"))
    assert per_ts["mean_e"].abs().max() < 1e-12
    assert per_ts["cov"].abs().max() < 1e-12


@pytest.mark.parametrize("frac", [0.4, 0.85, 0.999])
def test_causality_truncation(train: pd.DataFrame, frac: float) -> None:
    ts = train["timestamp"].drop_duplicates().sort_values()
    cut = ts.iloc[int(len(ts) * frac)]
    head = train[train["timestamp"] <= cut]
    for fn in (bar_vwap, close_location, close_location_residual):
        full = fn(train).loc[head.index].to_numpy()
        part = fn(head).to_numpy()
        np.testing.assert_array_equal(np.isnan(full), np.isnan(part))
        np.testing.assert_allclose(part, full, rtol=0, atol=1e-15, equal_nan=True)


# --- planted check (stretch) ----------------------------------------------------------------


def _planted_panel(reverting: bool, seed: int = 11, n_sym: int = 20, n_hours: int = 1500) -> pd.DataFrame:
    """
    Efficient log price m_t follows a random walk. The bar VWAP sits at m_t. The close is the VWAP
    plus a late-in-the-hour move e_t.

    reverting=True:  e_t is noise; the next bar's efficient price ignores it, so it reverts.
    reverting=False: e_t is real; the efficient price moves by e_t as well, so nothing reverts.
    """
    rng = np.random.default_rng(seed)
    m = np.zeros((n_hours, n_sym))
    e = rng.normal(0.0, 0.003, size=(n_hours, n_sym))
    shocks = rng.normal(0.0, 0.006, size=(n_hours, n_sym))
    for t in range(1, n_hours):
        m[t] = m[t - 1] + shocks[t] + (0.0 if reverting else e[t - 1])
    log_vwap = 4.0 + m
    log_close = log_vwap + e
    vol = rng.uniform(500.0, 1500.0, size=(n_hours, n_sym))
    ts = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    close = np.exp(log_close)
    vwap = np.exp(log_vwap)
    df = pd.DataFrame(
        {
            "timestamp": np.repeat(ts, n_sym),
            "symbol": np.tile([f"S{i:02d}" for i in range(n_sym)], n_hours),
            "close": close.ravel(),
            "high": np.maximum(close, vwap).ravel() * 1.001,
            "low": np.minimum(close, vwap).ravel() * 0.999,
            "volume": vol.ravel(),
            "quote_volume": (vol * vwap).ravel(),
        }
    )
    df["open"] = df.groupby("symbol")["close"].shift(1).fillna(df["close"])
    return df


def test_planted_reversion_separates_ic() -> None:
    ic_rev = information_coefficient(*_signal(_planted_panel(reverting=True)))
    ic_flat = information_coefficient(*_signal(_planted_panel(reverting=False)))
    assert ic_rev["mean_ic"] > 0.05 and ic_rev["t_stat"] > 5.0
    assert abs(ic_flat["mean_ic"]) < 0.02
    assert ic_rev["mean_ic"] - ic_flat["mean_ic"] > 0.05


def _signal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df, -close_location_residual(df)
