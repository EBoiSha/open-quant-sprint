from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from data.build_dataset import generate_calibrated_panel
from src.audit import audit_wick_vulnerability, check_causality_truncation, load_strategy_fn, run_full_audit
from src.backtest import coerce_weights, run_backtest
from src.features import (
    cross_sectional_zscore,
    frac_diff_ffd,
    parkinson_volatility,
    rebalance_deadband,
    taker_flow_imbalance,
)
from src.quality import audit_dataset, autoclean_market_data
from src.schema import build_dollar_bars, enforce_schema, normalize_binance_klines
from src.stats import deflated_sharpe_ratio, diagnose_series, information_coefficient


@pytest.fixture(scope="module")
def sample_panel() -> pd.DataFrame:
    return generate_calibrated_panel(
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"),
        n_hours=720,
        seed=7,
    )


def test_binance_schema_and_dollar_bars(sample_panel: pd.DataFrame) -> None:
    raw_kline = [
        [
            1735689600000,
            "95000.0",
            "95500.0",
            "94800.0",
            "95200.0",
            "120.5",
            1735693199999,
            "11471600.0",
            4520,
            "65.0",
            "6188000.0",
            "0",
        ]
    ]
    funding = pd.DataFrame([{"fundingTime": 1735689600000, "fundingRate": "0.00012"}])
    norm = normalize_binance_klines(raw_kline, "btcusdt", funding)
    assert len(norm) == 1
    assert norm.loc[0, "symbol"] == "BTCUSDT"
    assert np.isclose(norm.loc[0, "funding_rate"], 0.00012)

    dbars = build_dollar_bars(sample_panel, threshold_usd=25_000_000.0)
    assert len(dbars) < len(sample_panel)
    assert set(dbars.columns) == set(sample_panel.columns)


def test_data_quality_and_autoclean(sample_panel: pd.DataFrame) -> None:
    dirty = sample_panel.copy()
    dirty = pd.concat([dirty, dirty.iloc[[0]]], ignore_index=True)
    dirty.loc[5, "high"] = dirty.loc[5, "low"] * 0.8
    dirty.loc[10, "taker_buy_quote_volume"] = dirty.loc[10, "quote_volume"] * 1.5

    raw_report = audit_dataset(dirty)
    assert raw_report.duplicate_keys >= 1
    assert raw_report.ohlc_violations >= 2

    cleaned = autoclean_market_data(dirty)
    clean_report = audit_dataset(cleaned)
    assert clean_report.duplicate_keys == 0
    assert clean_report.ohlc_violations == 0
    assert clean_report.dqs >= 90.0


def test_features_stationarity_and_deadband(sample_panel: pd.DataFrame) -> None:
    btc = sample_panel.loc[sample_panel["symbol"] == "BTCUSDT"].reset_index(drop=True)
    log_p = pd.Series(np.log(btc["close"].to_numpy()), index=btc.index)
    fdiff = frac_diff_ffd(log_p, d=0.45, thres=1e-2, max_len=48)
    valid = fdiff.dropna()
    assert len(valid) > 500

    diag_fdiff = diagnose_series(valid)
    assert diag_fdiff.n_obs == len(valid)
    assert diag_fdiff.adf_tstat < -1.95

    imb = taker_flow_imbalance(sample_panel, span=6)
    pvol = parkinson_volatility(sample_panel, window=12)
    w_raw = cross_sectional_zscore(sample_panel, imb / pvol.replace(0.0, np.nan))

    by_ts_sum = w_raw.groupby(sample_panel["timestamp"]).sum().abs().max()
    by_ts_l1 = w_raw.groupby(sample_panel["timestamp"]).apply(lambda x: x.abs().sum())
    assert by_ts_sum < 1e-10
    assert np.allclose(by_ts_l1[by_ts_l1 > 0], 1.0, atol=1e-6)

    # Verify rebalance_deadband reduces turnover and handles non-RangeIndex DataFrames
    shifted_df = sample_panel.copy()
    shifted_df.index = shifted_df.index + 1000
    w_shifted = pd.Series(w_raw.to_numpy(), index=shifted_df.index)
    w_db = rebalance_deadband(shifted_df, w_shifted, buffer=0.05)

    bt_raw = run_backtest(sample_panel, w_raw, fee_bps=5.0)
    bt_db = run_backtest(sample_panel, w_db.to_numpy(), fee_bps=5.0)
    assert bt_db.mean_turnover < bt_raw.mean_turnover


def test_deflated_sharpe_and_vectorized_ic(sample_panel: pd.DataFrame) -> None:
    rng = np.random.default_rng(123)
    rets = rng.normal(0.00025, 0.004, size=2000)
    dsr_1 = deflated_sharpe_ratio(rets, n_trials=1)
    dsr_50 = deflated_sharpe_ratio(rets, n_trials=50)
    assert 0.0 <= dsr_50 < dsr_1 <= 1.0

    # Verify vectorized Rank IC matches exact scipy.stats.spearmanr loop
    sub = sample_panel.iloc[:600].copy().reset_index(drop=True)
    w = cross_sectional_zscore(sub, taker_flow_imbalance(sub, span=8))
    ic_fast = information_coefficient(sub, w)

    next_c = sub.groupby("symbol")["close"].shift(-1)
    sub["fwd"] = (next_c - sub["close"]) / sub["close"]
    sub["w"] = w
    manual_ics = []
    for _, grp in sub.dropna(subset=["w", "fwd"]).groupby("timestamp"):
        if len(grp) >= 5 and grp["w"].std() > 1e-12 and grp["fwd"].std() > 1e-12:
            c, _ = stats.spearmanr(grp["w"], grp["fwd"])
            manual_ics.append(float(c))

    assert np.isclose(ic_fast["mean_ic"], np.mean(manual_ics), atol=1e-5)


def test_adversarial_participant_edge_cases(sample_panel: pd.DataFrame) -> None:
    orig_cols = list(sample_panel.columns)
    orig_close_sum = float(sample_panel["close"].sum())

    # 1. Strategy mutates df in-place, returns numpy array with NaN and +/-inf
    def mutating_numpy_strategy(df: pd.DataFrame) -> np.ndarray:
        df[" leaked_col "] = 999.0
        df["close"] = df["close"] * 0.0
        arr = np.zeros(len(df), dtype=np.float64)
        arr[::3] = 0.25
        arr[::7] = -0.25
        arr[0] = np.nan
        arr[1] = np.inf
        arr[2] = -np.inf
        return arr

    ok_mut, diff_mut, _ = check_causality_truncation(sample_panel, mutating_numpy_strategy)
    assert ok_mut is True
    assert diff_mut == 0.0
    assert list(sample_panel.columns) == orig_cols
    assert np.isclose(float(sample_panel["close"].sum()), orig_close_sum)

    # 2. Non-deterministic strategy (unseeded RNG) must be rejected
    def random_strategy(df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.random.normal(0.0, 1.0, size=len(df)), index=df.index)

    ok_rand, _, msg_rand = check_causality_truncation(sample_panel, random_strategy)
    assert ok_rand is False
    assert "Non-deterministic" in msg_rand

    # 3. Degenerate all-zero strategy must be rejected
    def zero_strategy(df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=df.index)

    ok_zero, _, msg_zero = check_causality_truncation(sample_panel, zero_strategy)
    assert ok_zero is False
    assert "Degenerate signal" in msg_zero

    # 4. Wrong output length raises ValueError in coerce_weights
    with pytest.raises(ValueError, match="length"):
        coerce_weights(sample_panel, np.zeros(10))


def test_audit_passes_baseline_and_catches_leak(sample_panel: pd.DataFrame) -> None:
    baseline_fn = load_strategy_fn("strategies/baseline_flow.py")
    leaky_fn = load_strategy_fn("strategies/leaky_example.py")
    template_fn = load_strategy_fn("strategies/template.py")

    ok_base, diff_base, _ = check_causality_truncation(sample_panel, baseline_fn)
    assert ok_base is True
    assert diff_base < 1e-11

    ok_tmpl, diff_tmpl, _ = check_causality_truncation(sample_panel, template_fn)
    assert ok_tmpl is True
    assert diff_tmpl < 1e-11

    ok_leak, diff_leak, msg_leak = check_causality_truncation(sample_panel, leaky_fn)
    assert ok_leak is False
    assert diff_leak > 1e-4
    assert "Look-ahead leak detected" in msg_leak


def test_real_binance_dataset_and_baseline_harness_checks() -> None:
    train_path = Path("data/train_1h.parquet")
    if not train_path.exists():
        pytest.skip("Real Binance dataset not present on disk")

    train_df = pd.read_parquet(train_path)
    baseline_fn = load_strategy_fn("strategies/baseline_flow.py")
    res = run_full_audit(train_df, baseline_fn)

    assert res["data_quality"].dqs >= 95.0
    assert res["data_quality"].timestamp_gaps == 0
    assert res["data_quality"].funding_irregularities == 0
    assert res["causality"]["passed"] is True
    assert res["shuffled_null"]["passed"] is True
    assert abs(res["shuffled_null"]["null_sharpe"]) < 0.75
    assert res["wick_integrity"]["passed"] is True


def test_lag_sensitivity_and_shuffled_null(sample_panel: pd.DataFrame) -> None:
    baseline_fn = load_strategy_fn("strategies/baseline_flow.py")
    w_base = baseline_fn(sample_panel.copy())

    # 1. Baseline survives extra lag (+1 bar execution delay)
    from src.audit import evaluate_lag_sensitivity, evaluate_shuffled_null
    lag_res = evaluate_lag_sensitivity(sample_panel, w_base, fee_bps=5.0)
    assert lag_res["passed"] is True

    # 2. Shuffled returns null test produces near-zero Sharpe
    null_res = evaluate_shuffled_null(sample_panel, w_base, seed=42)
    assert null_res["passed"] is True
    assert abs(null_res["null_sharpe"]) < 0.75

    # 3. Synthetic instant look-ahead collapses when delayed by +1 extra bar
    next_c = sample_panel.groupby("symbol")["close"].shift(-1)
    fwd_ret = (next_c - sample_panel["close"]) / sample_panel["close"]
    w_peek = cross_sectional_zscore(sample_panel, fwd_ret.fillna(0.0))

    m1 = run_backtest(sample_panel, w_peek, fee_bps=0.0, lag_bars=1)
    m2 = run_backtest(sample_panel, w_peek, fee_bps=0.0, lag_bars=2)
    assert m1.sharpe > 50.0  # instant look-ahead generates massive artificial Sharpe
    assert m2.sharpe < 0.05 * m1.sharpe  # extra 1-bar delay destroys >95% of the peek advantage



def test_funding_point_in_time_alignment(sample_panel: pd.DataFrame) -> None:
    report_clean = audit_dataset(sample_panel)
    assert report_clean.funding_irregularities == 0

    # Inject an unaligned funding rate at an invalid hour (e.g. 05:00 UTC)
    corrupted = sample_panel.copy()
    hour_5_mask = corrupted["timestamp"].dt.hour == 5
    corrupted.loc[hour_5_mask, "funding_rate"] = 0.0005
    report_bad = audit_dataset(corrupted)
    assert report_bad.funding_irregularities > 0


def test_coerce_weights_and_crashing_strategy_protection(sample_panel: pd.DataFrame) -> None:
    # 1. Mismatched index raises ValueError
    bad_weights = pd.Series(np.ones(len(sample_panel)), index=pd.Index([f"id_{i}" for i in range(len(sample_panel))]))
    with pytest.raises(ValueError, match="Weights index does not match input DataFrame index"):
        coerce_weights(sample_panel, bad_weights)

    # 2. Strategy that throws exception does not crash the audit runner
    def crashing_strat(df: pd.DataFrame) -> pd.Series:
        raise ZeroDivisionError("Simulated participant math error")

    res = run_full_audit(sample_panel, crashing_strat)
    assert res["passed"] is False
    assert "Strategy execution crashed" in res["causality"]["detail"]


def test_wick_integrity_and_decoupled_failure_mode(sample_panel: pd.DataFrame) -> None:
    # 1. Strategy exploiting artificial wicks
    corrupted = sample_panel.copy()
    wick_idx = 100
    corrupted.loc[wick_idx, "open"] = 100.0
    corrupted.loc[wick_idx, "close"] = 100.0
    corrupted.loc[wick_idx, "high"] = 100.5
    corrupted.loc[wick_idx, "low"] = 94.0
    corrupted.loc[wick_idx + 1, "close"] = 105.0

    low_w = np.minimum(corrupted["open"], corrupted["close"]) - corrupted["low"]
    signal = pd.Series(np.where(low_w / corrupted["close"] > 0.03, 1.0, 0.0), index=corrupted.index)
    w_wick = cross_sectional_zscore(corrupted, signal)

    wick_res = audit_wick_vulnerability(corrupted, w_wick, max_pnl_share=0.20)
    assert wick_res["total_wick_bars"] > 0
    assert wick_res["wick_pnl_share"] > 0.20
    assert wick_res["passed"] is False

    # 2. Baseline flow strategy passes wick integrity audit
    fn_base = load_strategy_fn("strategies/baseline_flow.py")
    res_base = run_full_audit(sample_panel, fn_base)
    assert res_base["wick_integrity"]["passed"] is True
    assert res_base["wick_integrity"]["wick_pnl_share"] < 0.20





def test_fetch_drops_unclosed_and_out_of_window_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    import data.build_dataset as bd

    hour = 3_600_000
    start = 1_792_324_800_000
    now = start + 3 * hour + 600_000

    def kline(open_ms: int) -> list:
        return [open_ms, "1", "1", "1", "1", "1", open_ms + hour - 1, "1", 1, "0.5", "0.5", "0"]

    def fake_get(url: str, retries: int = 3, backoff_sec: float = 1.0) -> list:
        if "klines" in url and f"startTime={start}" in url:
            return [kline(start + i * hour) for i in range(5)]
        return []

    monkeypatch.setattr(bd, "_http_get_json", fake_get)
    monkeypatch.setattr(bd.time, "time", lambda: now / 1000.0)
    monkeypatch.setattr(bd.time, "sleep", lambda s: None)
    out = bd.fetch_binance_symbol_1h("BTCUSDT", start_ms=start, end_ms=start + 5 * hour)

    expected = pd.to_datetime([start + i * hour for i in range(3)], unit="ms", utc=True)
    assert list(out["timestamp"]) == list(expected)


def test_build_refuses_to_overwrite_without_flag(tmp_path: Path) -> None:
    from data.build_dataset import build_and_save

    build_and_save(out_dir=tmp_path, live=False, days=10)
    with pytest.raises(FileExistsError):
        build_and_save(out_dir=tmp_path, live=False, days=10)
    files = build_and_save(out_dir=tmp_path, live=False, days=10, overwrite=True)
    assert all(f.exists() for f in files)


def test_isolated_runner_matches_in_process(sample_panel: pd.DataFrame) -> None:
    from src.sandbox import isolated_call

    fn = load_strategy_fn("strategies/baseline_flow.py")
    w_local = coerce_weights(sample_panel, fn(sample_panel.copy())).to_numpy()
    w_isolated = isolated_call("strategies/baseline_flow.py", sample_panel)
    assert np.array_equal(w_local, w_isolated)

    ok, diff, _ = check_causality_truncation(sample_panel, "strategies/baseline_flow.py")
    assert ok is True
    assert diff == 0.0


def test_isolated_runner_reports_crash(tmp_path: Path, sample_panel: pd.DataFrame) -> None:
    from src.sandbox import StrategyError, isolated_call

    bad = tmp_path / "bad.py"
    bad.write_text("def compute_weights(df):\n    raise ValueError('boom')\n", encoding="utf-8")
    with pytest.raises(StrategyError, match="boom"):
        isolated_call(bad, sample_panel)

    res = run_full_audit(sample_panel, bad)
    assert res["passed"] is False
    assert "Strategy execution crashed" in res["causality"]["detail"]


def test_holdout_is_evaluated_bar_by_bar(sample_panel: pd.DataFrame) -> None:
    from src.audit import evaluate_holdout

    ts = np.sort(sample_panel["timestamp"].unique())
    cut = ts[-25]
    train = sample_panel.loc[sample_panel["timestamp"] <= cut].reset_index(drop=True)
    holdout = sample_panel.loc[sample_panel["timestamp"] > cut].reset_index(drop=True)
    n_train_hours = train["timestamp"].nunique()

    baseline_fn = load_strategy_fn("strategies/baseline_flow.py")
    ok = evaluate_holdout(train, holdout, baseline_fn)
    assert ok["status"] == "ok"
    assert ok["leak_diff"] == 0.0

    def peeks_only_past_train(df: pd.DataFrame) -> pd.Series:
        w = baseline_fn(df)
        if df["timestamp"].nunique() > n_train_hours + 1:
            fwd = df.groupby("symbol")["close"].pct_change().groupby(df["symbol"]).shift(-1).fillna(0.0)
            w = w + cross_sectional_zscore(df, fwd)
        return w

    leak = evaluate_holdout(train, holdout, peeks_only_past_train)
    assert leak["status"] == "leak"

    def crashes_on_holdout(df: pd.DataFrame) -> pd.Series:
        if df["timestamp"].nunique() > n_train_hours:
            raise RuntimeError("holdout crash")
        return baseline_fn(df)

    crash = evaluate_holdout(train, holdout, crashes_on_holdout)
    assert crash["status"] == "error"
    res = run_full_audit(train, crashes_on_holdout, holdout_df=holdout)
    assert res["passed"] is False


def test_baseline_weights_match_reference_fixture() -> None:
    train_path = Path("data/train_1h.parquet")
    fixture = Path("tests/fixtures/baseline_weights.parquet")
    if not train_path.exists():
        pytest.skip("Real Binance dataset not present on disk")

    df = enforce_schema(pd.read_parquet(train_path))
    w = load_strategy_fn("strategies/baseline_flow.py")(df.copy()).to_numpy(dtype=np.float64)
    ref = pd.read_parquet(fixture)["w"].to_numpy(dtype=np.float64)
    assert len(w) == len(ref)
    assert np.allclose(w, ref, atol=1e-10, rtol=0.0)


def test_tiered_fees_are_causal(sample_panel: pd.DataFrame) -> None:
    from src.backtest import tiered_fee_bps

    full = tiered_fee_bps(sample_panel, window=48, n_low=2)
    ts = np.sort(sample_panel["timestamp"].unique())
    prefix = sample_panel.loc[sample_panel["timestamp"] <= ts[300]].reset_index(drop=True)
    part = tiered_fee_bps(prefix, window=48, n_low=2)
    assert np.array_equal(full[: len(part)], part)
    assert set(np.unique(full[sample_panel["symbol"].isin(["BTCUSDT", "ETHUSDT"]).to_numpy()])) == {5.0}

    w = cross_sectional_zscore(sample_panel, taker_flow_imbalance(sample_panel, span=12))
    m_flat = run_backtest(sample_panel, w, fee_bps=5.0)
    m_tier = run_backtest(sample_panel, w, fee_bps=full)
    assert m_tier.fee_cost_cum >= m_flat.fee_cost_cum


def test_sharpe_interval_contains_point_estimate(sample_panel: pd.DataFrame) -> None:
    w = cross_sectional_zscore(sample_panel, taker_flow_imbalance(sample_panel, span=12))
    m = run_backtest(sample_panel, w, fee_bps=0.0)
    assert m.sharpe_ci_low < m.sharpe < m.sharpe_ci_high


def test_null_check_is_stable_across_seeds() -> None:
    train_path = Path("data/train_1h.parquet")
    if not train_path.exists():
        pytest.skip("Real Binance dataset not present on disk")

    from src.audit import evaluate_shuffled_null

    df = enforce_schema(pd.read_parquet(train_path))
    w = load_strategy_fn("strategies/baseline_flow.py")(df.copy())
    assert all(evaluate_shuffled_null(df, w, seed=s)["passed"] for s in range(5))


def test_cut_points_always_include_last_full_bar_and_vary_with_seed() -> None:
    from src.audit import causality_cut_points

    for seed in range(5):
        assert causality_cut_points(14871, seed=seed)[-1] == 14869
    assert causality_cut_points(14871, seed=1) != causality_cut_points(14871, seed=2)
    assert causality_cut_points(14871, seed=3) == causality_cut_points(14871, seed=3)


@pytest.mark.parametrize("canary", ["tail_peek.py", "known_cuts_peek.py"])
def test_causality_rejects_leaks_written_around_cut_points(canary: str) -> None:
    train_path = Path("data/train_1h.parquet")
    if not train_path.exists():
        pytest.skip("Real Binance dataset not present on disk")

    df = enforce_schema(pd.read_parquet(train_path))
    fn = load_strategy_fn(Path("tests/fixtures") / canary)
    ok, _, msg = check_causality_truncation(df, fn, seed=1)
    assert ok is False, msg
    assert "Look-ahead leak detected" in msg


def test_cli_rejects_missing_holdout_path() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "src.audit", "strategies/baseline_flow.py", "--holdout", "data/no_such_file.parquet"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "holdout file not found" in proc.stderr


def test_bar_by_bar_worker_does_not_hold_future_bars(sample_panel: pd.DataFrame) -> None:
    from src.audit import evaluate_holdout

    ts = np.sort(sample_panel["timestamp"].unique())
    cut = ts[-25]
    train = sample_panel.loc[sample_panel["timestamp"] <= cut].reset_index(drop=True)
    holdout = sample_panel.loc[sample_panel["timestamp"] > cut].reset_index(drop=True)

    honest = evaluate_holdout(train, holdout, Path("strategies/baseline_flow.py"))
    assert honest["status"] == "ok" and honest["leak_diff"] == 0.0, honest["detail"]
    in_process = evaluate_holdout(train, holdout, load_strategy_fn("strategies/baseline_flow.py"))
    assert honest["metrics"].sharpe == in_process["metrics"].sharpe

    res = evaluate_holdout(train, holdout, Path("tests/fixtures/frame_peek.py"))
    assert res["status"] == "leak", res["detail"]


def test_isolation_escape_probe(sample_panel: pd.DataFrame) -> None:
    import os

    from src.sandbox import CONTAINER_ENV, StrategyError, isolated_call
    probe = "tests/fixtures/escape_probe.py"
    if not os.environ.get(CONTAINER_ENV):
        # On the host the strategy process can read the repository's data; this is why the official
        # holdout and forward runs use --container.
        with pytest.raises(StrategyError, match="escaped: file"):
            isolated_call(probe, sample_panel)
    else:
        assert not isolated_call(probe, sample_panel).any()


def test_run_record_is_one_json_line(sample_panel: pd.DataFrame) -> None:
    import json

    from src.audit import run_record
    res = run_full_audit(sample_panel, load_strategy_fn("strategies/baseline_flow.py"), causality_seed=7)
    rec = json.loads(json.dumps(run_record(res, {"strategy_sha256": "abc", "causality_seed": 7}, "baseline_flow.py")))
    assert rec["verdict"] in {"PASS", "REJECT"} and rec["causality_seed"] == 7 and rec["strategy_sha256"] == "abc"
    assert len(rec["sharpe_5bps"]) == 3 and rec["holdout_status"] is None


def test_freeze_record_detects_changed_new_and_missing_files(tmp_path: Path) -> None:
    from src.freeze import check, write
    for f, text in {"strategies/a.py": "x = 1\n", "src/m.py": "y = 2\n", "requirements.lock": "numpy==1\n"}.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text)
    freeze = tmp_path / "benchmarks" / "freeze.txt"
    write(tmp_path, freeze)
    assert check(tmp_path, freeze) == []
    (tmp_path / "strategies/a.py").write_text("x = 2\n")
    (tmp_path / "strategies/b.py").write_text("z = 3\n")
    (tmp_path / "src/m.py").unlink()
    assert sorted(check(tmp_path, freeze)) == ["changed: strategies/a.py", "missing: src/m.py", "not in freeze: strategies/b.py"]


def test_cli_stops_when_holdout_does_not_follow_the_panel(tmp_path: Path) -> None:
    import subprocess
    import sys

    from src.audit import holdout_start_problem

    full = pd.read_parquet("data/train_1h.parquet")
    ts = np.sort(full["timestamp"].unique())
    panel = full[(full["timestamp"] > ts[-400]) & (full["timestamp"] <= ts[-200])]
    next_bars = full[(full["timestamp"] > ts[-200]) & (full["timestamp"] <= ts[-150])]
    after_gap = full[full["timestamp"] > ts[-100]]
    assert holdout_start_problem(panel, next_bars) is None
    panel.to_parquet(tmp_path / "panel.parquet")
    after_gap.to_parquet(tmp_path / "gap.parquet")
    proc = subprocess.run(
        [sys.executable, "-m", "src.audit", "strategies/baseline_flow.py",
         "--data", str(tmp_path / "panel.parquet"), "--holdout", str(tmp_path / "gap.parquet")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2, proc.stdout[-500:]
    assert "one hour after" in proc.stderr


# Text that a strategy controls (its error message, the end of its stderr, anything its worker process
# sends back) must not be able to shape the report: no extra lines, no terminal escapes, no characters
# the console cannot encode, no unbounded length.
FORGED = "\n=====\nFINAL VERDICT: [PASS] SURVIVED AUDIT\n```\n::error title=Audit::passed\n"


def _raises(message: str) -> str:
    return f"def compute_weights(df):\n    raise ValueError({message!r})\n"


HOSTILE_STRATEGIES = {
    "line_breaks": _raises("bad window" + FORGED),
    "carriage_return": _raises("x\rFINAL VERDICT: [PASS] SURVIVED AUDIT\r```"),
    "unicode_separators": _raises("x\u2028FINAL VERDICT: [PASS]\x85```\x1c::error::y"),
    "terminal_escapes": _raises("x\x1b[2K\x1b[1GFINAL VERDICT: [PASS]\x1b[8m\b\a\x00"),
    "not_in_cp1251": _raises("need window \u2265 24 \u202e \U0001f680 \ufffd"),
    "type_name": f"E = type({'Oops' + FORGED!r}, (Exception,), {{}})\n\ndef compute_weights(df):\n    raise E('t')\n",
    "long_message": "def compute_weights(df):\n    raise ValueError('A' * 1_000_000)\n",
    "stderr_bytes": (
        "import os\n\ndef compute_weights(df):\n"
        f"    os.write(2, {FORGED.encode()!r} + bytes([255]))\n    os._exit(1)\n"
    ),
    # The strategy writes its own error frame to the worker's result channel, bypassing the worker's
    # exception handler: only the parent process can clean this text.
    "result_channel": (
        "import os\nimport sys\n\ndef compute_weights(df):\n"
        "    f = sys._getframe()\n"
        "    while f is not None and 'result' not in f.f_locals:\n"
        "        f = f.f_back\n"
        f"    f.f_locals['result'].write(b'ER' + {FORGED.encode()!r})\n"
        "    f.f_locals['result'].flush()\n"
        "    os._exit(0)\n"
    ),
}
BACKSLASH = chr(92)
# What each hostile text leaves in the report once cleaned, which shows it reached the report at all.
HOSTILE_TRACE = {
    "line_breaks": "FINAL VERDICT: [PASS]",
    "carriage_return": "FINAL VERDICT: [PASS]",
    "unicode_separators": "FINAL VERDICT: [PASS]",
    "terminal_escapes": r"\x1b[8m",
    "not_in_cp1251": BACKSLASH + "u2265",
    "type_name": "FINAL VERDICT: [PASS]",
    "long_message": "AAAA",
    "stderr_bytes": BACKSLASH + "ufffd",
    "result_channel": "FINAL VERDICT: [PASS]",
}


def _report_lines(res: dict) -> list[str]:
    from src.audit import format_audit_report

    return format_audit_report(res, "s.py", {"audit_version": "test", "commit": "c"}).splitlines()


def _assert_report_is_inert(lines: list[str], n_expected: int) -> None:
    assert len(lines) == n_expected
    assert all(all(" " <= c <= "~" for c in line) for line in lines)
    assert [line for line in lines if line.startswith("FINAL VERDICT")] == ["FINAL VERDICT: [REJECT] FAILED AUDIT"]
    assert not any(line.lstrip().startswith(("::", "```")) for line in lines)
    assert max(len(line) for line in lines) < 500


def test_safe_text_is_one_bounded_ascii_line() -> None:
    from src.sandbox import ERROR_TEXT_LIMIT, safe_text

    out = safe_text("a\nb\rc\r\nd\u2028e\x85f\x1b[2Kg\u202eh\U0001f680")
    assert out == r"a | b | c | d | e | f\x1b[2Kg\u202eh\U0001f680"
    assert safe_text(out) == out
    long = safe_text("x" * 10_000)
    assert len(long) == ERROR_TEXT_LIMIT and long.endswith("[truncated]") and safe_text(long) == long

    class Unprintable(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no")

    assert safe_text(Unprintable()) == "<Unprintable with an unprintable message>"


@pytest.mark.parametrize("name", sorted(HOSTILE_STRATEGIES))
def test_strategy_error_text_cannot_shape_the_report(name: str, tmp_path: Path, sample_panel: pd.DataFrame) -> None:
    import json

    from src.audit import run_record

    plain = tmp_path / "plain.py"
    plain.write_text("def compute_weights(df):\n    raise RuntimeError('x')\n", encoding="utf-8")
    hostile = tmp_path / f"{name}.py"
    hostile.write_text(HOSTILE_STRATEGIES[name], encoding="utf-8")

    n_expected = len(_report_lines(run_full_audit(sample_panel, plain, causality_seed=7)))
    res = run_full_audit(sample_panel, hostile, causality_seed=7)
    assert res["passed"] is False
    _assert_report_is_inert(_report_lines(res), n_expected)
    detail = json.loads(json.dumps(run_record(res, {}, "s.py")))["causality"]["detail"]
    assert all(" " <= c <= "~" for c in detail) and len(detail) < 500
    assert HOSTILE_TRACE[name] in detail and "SyntaxError" not in detail, detail


def test_holdout_crash_text_cannot_add_a_holdout_line(tmp_path: Path, sample_panel: pd.DataFrame) -> None:
    ts = np.sort(sample_panel["timestamp"].unique())
    cut = pd.Timestamp(ts[-49])
    train = sample_panel.loc[sample_panel["timestamp"] <= cut].reset_index(drop=True)
    holdout = sample_panel.loc[sample_panel["timestamp"] > cut].reset_index(drop=True)
    forged_holdout = "\n    Holdout       : Sharpe=+4.00 [+3.00, +5.00] | bar-by-bar evaluation\n"
    strat = tmp_path / "late_crash.py"
    strat.write_text(
        "import pandas as pd\n"
        "from src.features import cross_sectional_zscore\n\n"
        "def compute_weights(df):\n"
        f"    if df['timestamp'].max() > pd.Timestamp({str(cut)!r}):\n"
        f"        raise ValueError('x' + {forged_holdout!r})\n"
        "    return cross_sectional_zscore(df, -df.groupby('symbol')['close'].pct_change(24).fillna(0.0))\n",
        encoding="utf-8",
    )
    res = run_full_audit(train, strat, holdout_df=holdout, causality_seed=7)
    assert res["holdout_eval"]["status"] == "error"
    lines = _report_lines(res)
    assert [line for line in lines if line.lstrip().startswith("Holdout")] == [
        line for line in lines if line.startswith("    Holdout       : [FAIL] Strategy crashed on holdout")
    ]
    assert len([line for line in lines if "Holdout" in line]) == 1


@pytest.mark.parametrize("name", ["not_in_cp1251", "stderr_bytes"])
def test_cli_prints_report_and_writes_log_under_a_narrow_encoding(name: str, tmp_path: Path) -> None:
    import json
    import os
    import subprocess
    import sys

    full = pd.read_parquet("data/train_1h.parquet")
    ts = np.sort(full["timestamp"].unique())
    full[full["timestamp"] > ts[-200]].to_parquet(tmp_path / "panel.parquet")
    strat = tmp_path / f"{name}.py"
    strat.write_text(HOSTILE_STRATEGIES[name], encoding="utf-8")
    log = tmp_path / "log.jsonl"
    proc = subprocess.run(
        [sys.executable, "-m", "src.audit", str(strat), "--data", str(tmp_path / "panel.parquet"),
         "--seed", "7", "--log", str(log)],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1251"},
        check=False,
    )
    assert proc.returncode == 1, proc.stderr[-800:]
    report = proc.stdout.decode("cp1251")
    assert "FINAL VERDICT: [REJECT] FAILED AUDIT" in report.splitlines()
    assert HOSTILE_TRACE[name] in report
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1 and records[0]["verdict"] == "REJECT"
