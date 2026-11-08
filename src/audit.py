from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tokenize
from typing import Callable
import numpy as np
import pandas as pd

from src.backtest import BacktestMetrics, coerce_weights, run_backtest, tiered_fee_bps
from src.quality import QualityReport, audit_dataset
from src.sandbox import CONTAINER_ENV, isolated_call, isolated_call_timed, isolated_stream, load_module_fn, safe_text, stream_weights
from src.schema import enforce_schema
from src.stats import information_coefficient

AUDIT_VERSION = "1.3.2"

StrategyFn = Callable[[pd.DataFrame], pd.Series]
Strategy = StrategyFn | str | Path


def load_strategy_fn(file_path: str | Path) -> StrategyFn:
    return load_module_fn(file_path)


def _caller(strategy: Strategy) -> Callable[[pd.DataFrame], np.ndarray]:
    """
    A file path runs every call in a fresh interpreter (no state shared between calls, no access to the
    repository working directory). A callable runs in-process and is meant for unit tests.
    """
    if isinstance(strategy, (str, Path)):
        return lambda d: isolated_call(strategy, d)
    return lambda d: coerce_weights(d, strategy(d.copy())).to_numpy(dtype=np.float64)


def _timed_call(strategy: Strategy, df: pd.DataFrame) -> tuple[np.ndarray, float]:
    if isinstance(strategy, (str, Path)):
        return isolated_call_timed(strategy, df)
    t0 = time.perf_counter()
    w = coerce_weights(df, strategy(df.copy())).to_numpy(dtype=np.float64)
    return w, time.perf_counter() - t0


def _streamer(strategy: Strategy) -> Callable[[pd.DataFrame, list], np.ndarray]:
    if isinstance(strategy, (str, Path)):
        return lambda d, ts: isolated_stream(strategy, d, ts)
    return lambda d, ts: stream_weights(strategy, d, ts)


def new_causality_seed() -> int:
    return int(np.random.default_rng().integers(0, 2**32))


def causality_cut_points(
    n_timestamps: int,
    checkpoints: tuple[float, ...] = (0.4, 0.65, 0.85, 0.95, 0.99),
    n_random: int = 8,
    seed: int = 0,
) -> list[int]:
    rng = np.random.default_rng(seed)
    fracs = list(checkpoints) + list(rng.uniform(0.30, 0.995, size=n_random))
    cuts = {min(n_timestamps - 2, max(10, int(n_timestamps * f))) for f in fracs}
    # The last full bar is always a cut, so no rows except the final one (which earns no return)
    # escape comparison with a prefix run.
    cuts.add(n_timestamps - 2)
    return sorted(cuts)


def check_causality_truncation(
    df: pd.DataFrame,
    strategy: Strategy,
    checkpoints: tuple[float, ...] = (0.4, 0.65, 0.85, 0.95, 0.99),
    n_random_cuts: int = 8,
    tol: float = 1e-9,
    full_weights: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[bool, float, str]:
    """
    1. Determinism: two separate runs on the same prefix must give identical weights.
    2. Causality: weights computed on each prefix [0..T_k] must equal the full-run weights at timestamps <= T_k.
    Prefixes run in increasing order and before the full run, each in its own process for file strategies,
    so nothing computed on later data can be reused for earlier rows.
    Random cuts use a fresh seed per run unless one is given, so a strategy cannot be written around them;
    the seed is reported so a verdict can be reproduced with --seed.
    """
    if seed is None:
        seed = new_causality_seed()
    call = _caller(strategy)
    full_df = df.copy()
    unique_ts = np.sort(full_df["timestamp"].unique())
    cuts = causality_cut_points(len(unique_ts), checkpoints, n_random_cuts, seed)

    prefix_results = []
    for idx, cut_idx in enumerate(cuts):
        t_cut = unique_ts[cut_idx]
        mask = (full_df["timestamp"] <= t_cut).to_numpy()
        sub_df = full_df.loc[mask].reset_index(drop=True)
        try:
            w_sub = call(sub_df)
        except Exception as exc:
            return False, float("inf"), f"Strategy crashed on prefix cut at {t_cut}: {safe_text(exc)}"

        if idx == 0:
            try:
                w_repeat = call(sub_df)
            except Exception as exc:
                return False, float("inf"), f"Strategy crashed on repeated execution: {safe_text(exc)}"
            det_diff = float(np.max(np.abs(w_sub - w_repeat)))
            if not det_diff <= tol:
                return False, det_diff, f"Non-deterministic strategy output (diff={det_diff:.3e} on identical input)"
        prefix_results.append((t_cut, mask, w_sub))

    if full_weights is None:
        try:
            full_weights = call(full_df)
        except Exception as exc:
            return False, float("inf"), f"Strategy crashed during execution: {safe_text(exc)}"

    if np.all(np.abs(full_weights) < 1e-12):
        return False, 0.0, "Degenerate signal: strategy produced all-zero weights"

    max_diff = 0.0
    for t_cut, mask, w_sub in prefix_results:
        w_ref = full_weights[mask]
        if len(w_sub) != len(w_ref):
            return False, float("inf"), f"Row count mismatch on truncated slice at {t_cut}"
        diff = float(np.max(np.abs(w_sub - w_ref)))
        if np.isnan(diff):
            return False, float("nan"), f"NaN diff encountered at {t_cut}"
        max_diff = max(max_diff, diff)
        if diff > tol:
            return (
                False,
                max_diff,
                f"Look-ahead leak detected at {t_cut}: max|w_trunc - w_full| = {diff:.3e} > {tol:.1e} (seed={seed})",
            )

    return (
        True,
        max_diff,
        f"Causal invariance verified across {len(prefix_results)} prefix cuts (max diff={max_diff:.2e}, seed={seed})",
    )


def purged_oos_split_eval(
    df: pd.DataFrame,
    weights: pd.Series,
    split_frac: float = 0.70,
    embargo_bars: int = 24,
    fee_bps: float = 5.0,
) -> dict[str, BacktestMetrics]:
    """
    Splits timeline into In-Sample (IS) and Out-of-Sample (OOS) with an embargo buffer
    to prevent serial correlation spillover across the boundary.
    """
    unique_ts = np.sort(df["timestamp"].unique())
    n_ts = len(unique_ts)
    split_idx = int(n_ts * split_frac)
    oos_start_idx = min(n_ts - 1, split_idx + embargo_bars)

    t_is_end = unique_ts[split_idx - 1]
    t_oos_start = unique_ts[oos_start_idx]

    is_mask = df["timestamp"] <= t_is_end
    oos_mask = df["timestamp"] >= t_oos_start

    df_is = df.loc[is_mask].reset_index(drop=True)
    w_is = pd.Series(weights.loc[is_mask].to_numpy(), index=df_is.index)

    df_oos = df.loc[oos_mask].reset_index(drop=True)
    w_oos = pd.Series(weights.loc[oos_mask].to_numpy(), index=df_oos.index)

    return {
        "IS": run_backtest(df_is, w_is, fee_bps=fee_bps),
        "OOS": run_backtest(df_oos, w_oos, fee_bps=fee_bps),
    }


def evaluate_lag_sensitivity(
    df: pd.DataFrame,
    weights: pd.Series,
    fee_bps: float = 5.0,
) -> dict:
    """
    Stress-tests alpha persistence by comparing 1-bar execution delay against 2-bar delay.
    A genuine structural signal decays smoothly; a look-ahead or instant-close signal collapses.
    """
    m1 = run_backtest(df, weights, fee_bps=fee_bps, lag_bars=1)
    m2 = run_backtest(df, weights, fee_bps=fee_bps, lag_bars=2)

    sr1 = m1.sharpe
    sr2 = m2.sharpe
    retention = (sr2 / sr1) if sr1 > 1e-6 else (1.0 if abs(sr1 - sr2) < 1e-6 else 0.0)
    passed = bool(sr2 > 0.0 if sr1 > 0.0 else sr2 >= sr1 - 0.5)

    return {
        "sr_lag1": round(sr1, 2),
        "sr_lag2": round(sr2, 2),
        "retention_rate": round(retention, 3),
        "passed": passed,
        "m_lag1": m1,
        "m_lag2": m2,
    }


def evaluate_shuffled_null(
    df: pd.DataFrame,
    weights: pd.Series,
    seed: int = 42,
    n_perm: int = 200,
    block_hours: int = 24,
    max_abs_mean_sr: float = 0.75,
) -> dict:
    """
    Timing null. Daily blocks of the return panel are permuted in time (the same permutation for every
    symbol) while the held weights stay in place. Each asset keeps its total return over the sample, so
    the null mean measures how much of the Sharpe a strategy would earn from a static cross-sectional
    tilt alone; a pipeline that misaligns weights and returns shows up here as well.
    The check fails when the mean of the n_perm null Sharpe ratios exceeds max_abs_mean_sr in absolute value.
    The p-value of the actual gross Sharpe against the null is reported for information only:
    it is computed on data the strategy was developed on and is not a significance test.
    """
    w_clean = coerce_weights(df, weights)
    work = df[["timestamp", "symbol", "close"]].copy()
    work["w_held"] = w_clean.groupby(df["symbol"]).shift(1).fillna(0.0).to_numpy()
    work["ret"] = work.groupby("symbol")["close"].pct_change().fillna(0.0).to_numpy()

    w_mat = work.pivot_table(index="timestamp", columns="symbol", values="w_held", aggfunc="sum").fillna(0.0)
    r_mat = work.pivot_table(index="timestamp", columns="symbol", values="ret", aggfunc="sum").fillna(0.0)
    r_mat = r_mat.reindex(index=w_mat.index, columns=w_mat.columns).fillna(0.0)
    w_arr = w_mat.to_numpy()
    r_arr = r_mat.to_numpy()
    n_t = len(w_arr)

    def ann_sr(pnl: np.ndarray) -> float:
        sd = float(np.std(pnl, ddof=1))
        return float(np.mean(pnl) / sd * np.sqrt(8760)) if sd > 1e-12 else 0.0

    actual_sr = ann_sr((w_arr * r_arr).sum(axis=1))

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_t / block_hours))
    block_idx = np.arange(n_blocks * block_hours).reshape(n_blocks, block_hours)
    null_srs = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        order = block_idx[rng.permutation(n_blocks)].ravel()
        order = order[order < n_t]
        null_srs[k] = ann_sr((w_arr * r_arr[order]).sum(axis=1))

    mean_sr = float(np.mean(null_srs))
    p_value = float((1 + np.sum(null_srs >= actual_sr)) / (n_perm + 1))
    return {
        "null_sharpe": round(mean_sr, 2),
        "null_sd": round(float(np.std(null_srs, ddof=1)), 2),
        "actual_gross_sharpe": round(actual_sr, 2),
        "p_value": round(p_value, 4),
        "n_perm": n_perm,
        "passed": bool(abs(mean_sr) < max_abs_mean_sr),
    }


def audit_wick_vulnerability(
    df: pd.DataFrame,
    weights: pd.Series,
    max_pnl_share: float = 0.20,
    wick_ratio_threshold: float = 3.5,
    min_wick_pct: float = 0.015,
) -> dict:
    """
    Evaluates whether strategy alpha is disproportionately harvested from
    unfillable liquidation wicks or transient bad marks.
    Calculates the fraction of positive gross PnL earned on bars immediately
    following extreme wick / price dislocation bars.
    """
    work = df[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
    w_clean = coerce_weights(df, weights)
    work["w_held"] = w_clean.groupby(df["symbol"]).shift(1).fillna(0.0)
    work["ret"] = work.groupby("symbol")["close"].pct_change().fillna(0.0)
    work["gross_pnl"] = work["w_held"] * work["ret"]

    open_p = work["open"].to_numpy(dtype=np.float64)
    high_p = work["high"].to_numpy(dtype=np.float64)
    low_p = work["low"].to_numpy(dtype=np.float64)
    close_p = work["close"].to_numpy(dtype=np.float64)

    upper_wick = high_p - np.maximum(open_p, close_p)
    lower_wick = np.minimum(open_p, close_p) - low_p
    body = np.abs(close_p - open_p)
    max_wick = np.maximum(upper_wick, lower_wick)

    safe_body = np.where(body > 1e-6, body, 1e-6 * np.maximum(1e-6, close_p))
    wick_ratio = max_wick / safe_body
    wick_pct = max_wick / np.maximum(1e-6, close_p)

    is_wick_bar = (wick_ratio > wick_ratio_threshold) & (wick_pct > min_wick_pct)
    work["entered_on_wick"] = (
        pd.Series(is_wick_bar, index=df.index)
        .groupby(df["symbol"])
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    pos_pnl = np.maximum(0.0, work["gross_pnl"].to_numpy(dtype=np.float64))
    total_pos_pnl = float(np.sum(pos_pnl))
    wick_pos_pnl = float(np.sum(pos_pnl[work["entered_on_wick"].to_numpy(dtype=bool)]))

    pnl_share = (wick_pos_pnl / total_pos_pnl) if total_pos_pnl > 1e-6 else 0.0
    passed = bool(pnl_share <= max_pnl_share)

    return {
        "wick_pnl_share": round(pnl_share, 4),
        "total_wick_bars": int(np.sum(is_wick_bar)),
        "passed": passed,
    }


def holdout_start_problem(panel: pd.DataFrame, holdout: pd.DataFrame) -> str | None:
    """
    The holdout or forward file has to start one hour after the panel ends. With a gap, rolling
    features at the start of the period would quietly run on stale bars; with an overlap, the keys clash.
    """
    expected = panel["timestamp"].max() + pd.Timedelta(hours=1)
    start = holdout["timestamp"].min()
    if start == expected:
        return None
    return f"holdout starts at {start}, expected {expected}, one hour after the --data panel ends"


def evaluate_holdout(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    strategy: Strategy,
    tol: float = 1e-9,
) -> dict:
    """
    Holdout weights are computed bar by bar: the weight at hour t comes from compute_weights on
    train + holdout rows up to t. A one-shot run on the combined panel is compared against that
    stream; any difference means the strategy used holdout bars that were not yet available.
    """
    h_clean = enforce_schema(holdout_df)
    combined = enforce_schema(pd.concat([train_df, h_clean], ignore_index=True))
    if combined.duplicated(subset=["timestamp", "symbol"]).any():
        return {"status": "error", "detail": "Holdout overlaps the training panel (duplicate timestamp/symbol keys)"}

    h_start = h_clean["timestamp"].min()
    h_mask = (combined["timestamp"] >= h_start).to_numpy()
    h_timestamps = list(np.sort(combined.loc[h_mask, "timestamp"].unique()))

    try:
        w_stream = _streamer(strategy)(combined, h_timestamps)
        w_full = _caller(strategy)(combined)[h_mask]
    except Exception as exc:
        return {"status": "error", "detail": f"Strategy crashed on holdout: {safe_text(exc)}"}

    if len(w_stream) != int(h_mask.sum()):
        return {"status": "error", "detail": "Holdout weight count mismatch"}

    leak_diff = float(np.max(np.abs(w_stream - w_full))) if len(w_stream) else 0.0
    h_slice = combined.loc[h_mask].reset_index(drop=True)
    metrics = run_backtest(h_slice, pd.Series(w_stream, index=h_slice.index), fee_bps=5.0)
    if not leak_diff <= tol:
        return {
            "status": "leak",
            "detail": f"Holdout look-ahead: one-shot weights differ from bar-by-bar weights (max diff={leak_diff:.3e})",
            "metrics": metrics,
            "leak_diff": leak_diff,
        }
    return {"status": "ok", "detail": "bar-by-bar evaluation", "metrics": metrics, "leak_diff": leak_diff}


def run_full_audit(
    df: pd.DataFrame,
    strategy: Strategy,
    fee_grid_bps: tuple[float, ...] = (0.0, 3.0, 5.0, 10.0),
    holdout_df: pd.DataFrame | None = None,
    causality_seed: int | None = None,
    n_random_cuts: int = 8,
) -> dict:
    df_clean = enforce_schema(df)
    dq: QualityReport = audit_dataset(df_clean)

    try:
        w_arr, elapsed_s = _timed_call(strategy, df_clean)
        weights = pd.Series(w_arr, index=df_clean.index)
    except Exception as exc:
        empty_m = BacktestMetrics(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0]))
        return {
            "passed": False,
            "data_quality": dq,
            "causality": {"passed": False, "max_diff": float("inf"), "detail": f"Strategy execution crashed: {safe_text(exc)}"},
            "ic": {"mean_ic": 0.0, "ic_ir": 0.0, "t_stat": 0.0, "p_value": 1.0, "n_obs": 0},
            "fee_curve": {bps: empty_m for bps in fee_grid_bps},
            "split": {"IS": empty_m, "OOS": empty_m},
            "lag_sensitivity": {"passed": False, "sr_lag1": 0.0, "sr_lag2": 0.0, "retention_rate": 0.0},
            "shuffled_null": {"passed": False, "null_sharpe": 0.0, "null_sd": 0.0, "p_value": 1.0, "n_perm": 0},
            "wick_integrity": {"passed": False, "wick_pnl_share": 1.0, "total_wick_bars": 0},
            "holdout": None,
            "holdout_eval": None,
            "perf": {"elapsed_ms": 0.0, "rows_per_sec": 0},
        }

    elapsed_ms = elapsed_s * 1000.0
    rows_per_sec = len(df_clean) / max(1e-6, (elapsed_ms / 1000.0))

    causal_ok, max_leak_diff, causal_msg = check_causality_truncation(
        df_clean,
        strategy,
        n_random_cuts=n_random_cuts,
        full_weights=weights.to_numpy(dtype=np.float64),
        seed=causality_seed,
    )
    ic_stats = information_coefficient(df_clean, weights)

    fee_curve: dict = {bps: run_backtest(df_clean, weights, fee_bps=bps) for bps in fee_grid_bps}
    fee_curve["tiered"] = run_backtest(df_clean, weights, fee_bps=tiered_fee_bps(df_clean))
    split_res = purged_oos_split_eval(df_clean, weights, fee_bps=5.0)
    lag_res = evaluate_lag_sensitivity(df_clean, weights, fee_bps=5.0)
    null_res = evaluate_shuffled_null(df_clean, weights, seed=42)
    wick_res = audit_wick_vulnerability(df_clean, weights)

    holdout_eval = None
    if holdout_df is not None:
        if holdout_df.empty:
            holdout_eval = {"status": "error", "detail": "Holdout file is empty"}
        else:
            holdout_eval = evaluate_holdout(df_clean, holdout_df, strategy)
    holdout_metrics = holdout_eval.get("metrics") if holdout_eval else None

    m5 = fee_curve.get(5.0, fee_curve[fee_grid_bps[0]])
    passed = bool(
        causal_ok
        and dq.dqs >= 80.0
        and split_res["OOS"].sharpe > 0.0
        and m5.sharpe > 0.0
        and m5.mean_turnover > 1e-5
        and lag_res["passed"]
        and null_res["passed"]
        and wick_res["passed"]
        and (holdout_eval is None or holdout_eval["status"] == "ok")
    )

    return {
        "passed": passed,
        "data_quality": dq,
        "causality": {"passed": causal_ok, "max_diff": max_leak_diff, "detail": causal_msg},
        "ic": ic_stats,
        "fee_curve": fee_curve,
        "split": split_res,
        "lag_sensitivity": lag_res,
        "shuffled_null": null_res,
        "wick_integrity": wick_res,
        "holdout": holdout_metrics,
        "holdout_eval": holdout_eval,
        "perf": {"elapsed_ms": round(elapsed_ms, 2), "rows_per_sec": int(rows_per_sec)},
        "weights": weights,
    }


def _ci(m: BacktestMetrics) -> str:
    if np.isnan(m.sharpe_ci_low):
        return "n/a"
    flag = "" if (m.sharpe_ci_low > 0.0 or m.sharpe_ci_high < 0.0) else " (includes 0)"
    return f"[{m.sharpe_ci_low:+.2f}, {m.sharpe_ci_high:+.2f}]{flag}"


def format_audit_report(res: dict, strategy_name: str, provenance: dict | None = None) -> str:
    dq: QualityReport = res["data_quality"]
    c = res["causality"]
    ic = res["ic"]
    split = res["split"]
    perf = res["perf"]
    lag = res["lag_sensitivity"]
    null = res["shuffled_null"]
    wick = res.get("wick_integrity", {"passed": True, "wick_pnl_share": 0.0, "total_wick_bars": 0})
    m5 = res["fee_curve"].get(5.0, next(iter(res["fee_curve"].values())))

    lines = [
        "=" * 72,
        f"SYNDICATE AUDIT REPORT :: {strategy_name}",
        "=" * 72,
    ]
    if provenance:
        lines.append(
            f"    audit v{provenance.get('audit_version', AUDIT_VERSION)} | commit {provenance.get('commit', 'n/a')} | "
            f"strategy sha256 {provenance.get('strategy_sha256', 'n/a')}"
        )
        lines.append(
            f"    data sha256 {provenance.get('data_sha256', 'n/a')}"
            + (f" | holdout sha256 {provenance['holdout_sha256']}" if provenance.get("holdout_sha256") else "")
        )
        if provenance.get("causality_seed") is not None:
            lines.append(
                f"    causality seed {provenance['causality_seed']} | random cuts {provenance.get('n_random_cuts', 'n/a')} "
                f"(reproduce with --seed {provenance['causality_seed']})"
            )
        if provenance.get("isolation"):
            lines.append(f"    strategy process: {provenance['isolation']}")
    lines += [
        f"[1] DATA QUALITY     : DQS={dq.dqs}/100 ({dq.status}) | rows={dq.total_rows} | symbols={dq.n_symbols} | gaps={dq.timestamp_gaps} | funding_irreg={dq.funding_irregularities}",
        f"[2] CAUSALITY CHECK  : {'[PASS]' if c['passed'] else '[FAIL]'} {c['detail']}",
        f"[3] RANK IC (1h fwd) : mean={ic['mean_ic']:+.5f} | IC_IR={ic['ic_ir']:+.3f} | t={ic['t_stat']:+.2f} (p={ic['p_value']:.4f})",
        f"[4] RUNTIME          : {perf['elapsed_ms']:.1f} ms ({perf['rows_per_sec']:,} rows/sec)",
        "-" * 72,
        "[5] TRANSACTION COST DECAY MATRIX (1-bar lag + funding included):",
        f"    {'Fee (bps)':<12} {'Sharpe':>8} {'95% CI (block bootstrap)':>28} {'DSR(N=10)':>10} {'Ann.Ret':>9} {'MaxDD':>9} {'Turnover':>9}",
    ]
    for bps, m in res["fee_curve"].items():
        label = f"tiered ~{m.fee_bps:.1f}" if bps == "tiered" else f"{bps:>6.1f} bps"
        lines.append(
            f"    {label:<12} {m.sharpe:>+8.2f} {_ci(m):>28} {m.deflated_sharpe:>10.3f} "
            f"{m.ann_return * 100:>+8.2f}% {m.max_drawdown * 100:>+8.2f}% {m.mean_turnover:>9.4f}"
        )
    lines.append("    tiered = BTC/ETH 5 bps, 6 least liquid by trailing 30d volume 10 bps, others 7 bps")

    is_m = split["IS"]
    oos_m = split["OOS"]
    lines.extend(
        [
            "-" * 72,
            "[6] PURGED SPLIT (70% In-Sample / 24h Embargo / 30% Out-of-Sample @ 5.0 bps):",
            f"    In-Sample     : Sharpe={is_m.sharpe:+.2f} {_ci(is_m)} | DSR={is_m.deflated_sharpe:.3f} | Ann.Ret={is_m.ann_return * 100:+.2f}% | MaxDD={is_m.max_drawdown * 100:+.2f}%",
            f"    Out-of-Sample : Sharpe={oos_m.sharpe:+.2f} {_ci(oos_m)} | DSR={oos_m.deflated_sharpe:.3f} | Ann.Ret={oos_m.ann_return * 100:+.2f}% | MaxDD={oos_m.max_drawdown * 100:+.2f}%",
        ]
    )
    h_eval = res.get("holdout_eval")
    if h_eval is not None:
        if h_eval.get("metrics") is not None and h_eval["status"] == "ok":
            hm: BacktestMetrics = h_eval["metrics"]
            lines.append(
                f"    Holdout       : Sharpe={hm.sharpe:+.2f} {_ci(hm)} | Ann.Ret={hm.ann_return * 100:+.2f}% | MaxDD={hm.max_drawdown * 100:+.2f}% | {h_eval['detail']}"
            )
        else:
            lines.append(f"    Holdout       : [FAIL] {h_eval['detail']}")

    lag_status = "PASS" if lag["passed"] else ("FAIL - microstructure decay / half-life < 1h" if c["passed"] else "FAIL - lookahead leak")

    lines.extend(
        [
            "-" * 72,
            "[7] LAG SENSITIVITY TEST (+1 extra bar execution lag @ 5.0 bps):",
            f"    1-bar lag SR  : {lag['sr_lag1']:+.2f} | 2-bar lag SR: {lag['sr_lag2']:+.2f} | Retention: {lag['retention_rate'] * 100:.1f}% [{lag_status}]",
            f"[8] NULL PIPELINE CHECK ({null.get('n_perm', 0)} daily-block permutations of returns):",
            f"    Null SR mean  : {null['null_sharpe']:+.2f} (sd {null.get('null_sd', 0.0):.2f}, hurdle |mean| < 0.75) [{'PASS' if null['passed'] else 'FAIL - static tilt or pipeline artifact'}]",
            f"    Gross SR vs null: p={null.get('p_value', 1.0):.4f} (in-sample, informational only)",
            "[9] FRICTION DRAG BREAKDOWN (@ 5.0 bps taker fee):",
            f"    Gross PnL     : {m5.gross_pnl_cum * 100:+.2f}%",
            f"    Fee Cost      : {m5.fee_cost_cum * 100:+.2f}%",
            f"    Funding Drag  : {m5.funding_cost_cum * 100:+.2f}%",
            f"    Net PnL       : {m5.net_pnl_cum * 100:+.2f}%",
            "[10] WICK & MARK INTEGRITY (liquidation wick / bad mark hurdle <= 20.0%):",
            f"    Wick PnL Share: {wick['wick_pnl_share'] * 100:.1f}% ({wick['total_wick_bars']} anomaly bars) [{'PASS - structural flow' if wick['passed'] else 'FAIL - bad mark exploitation'}]",
            "-" * 72,
            "[11] EXECUTION REALISM & MODEL BOUNDARIES:",
            "    Fills         : signal on the close of bar t, filled at that close; lag-2 test = one hour of real delay",
            "    Costs         : flat taker fee grid + tiered row, snapshot funding on the position held across settlement",
            "    Scope Caveat  : Excludes orderbook queueing, depth-scaled impact (Almgren-Chriss), & adverse selection",
        ]
    )
    corr = res.get("baseline_corr")
    if corr is not None:
        lines.append(f"[12] CORRELATION    : hourly net returns vs baseline_flow.py @ 5 bps = {corr:+.2f}")
    flags = res.get("review_flags")
    if flags is not None:
        lines.append(f"[13] REVIEW FLAGS   : {'; '.join(flags) if flags else 'none'} (manual review, not part of the verdict)")
    lines.extend(
        [
            "=" * 72,
            f"FINAL VERDICT: {'[PASS] SURVIVED AUDIT' if res['passed'] else '[REJECT] FAILED AUDIT'}",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


REVIEW_PATTERNS = {
    "file or path access": r"\bopen\(|read_(parquet|csv|table|feather|pickle|json|excel|hdf)|\bnp\.(load|fromfile)|\bPath\(|pathlib|__file__",
    "dynamic import or eval": r"__import__|importlib|\beval\(|\bexec\(|\bcompile\(|__builtins__|getattr\(",
    "system or network": r"^\s*(import|from)\s+(os|sys|socket|subprocess|urllib|requests|http|pickle|shelve)\b",
    "harness internals": r"^\s*(import|from)\s+src\.(audit|backtest|sandbox)\b",
    "negative shift or centered window": r"shift\(\s*-|center\s*=\s*True",
    "module-level mutable state": r"^\s*global\s|^_?\w+\s*=\s*(\{\}|\[\]|dict\(|list\(|set\()",
    "branching on data length": r"(nunique\(\)|len\(\w+\)|\.max\(\))\s*[<>]=?",
}


def review_flags(path: Path) -> list[str]:
    """Patterns that need a human look in a strategy file. Reported only; never part of the verdict."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    code_lines = text.splitlines()
    try:
        kept: dict[int, list[str]] = {}
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT) or tok.type >= tokenize.N_TOKENS:
                continue
            kept.setdefault(tok.start[0], []).append(tok.string if tok.type != tokenize.INDENT else "")
        code_lines = [" ".join(kept.get(i + 1, [])) for i in range(len(code_lines))]
    except (tokenize.TokenError, SyntaxError):
        pass
    code_lines = [re.sub(r"\s*([().,=\[\]{}<>-])\s*", r"\1", line) for line in code_lines]
    found = []
    for label, pattern in REVIEW_PATTERNS.items():
        lines = [i + 1 for i, line in enumerate(code_lines) if re.search(pattern, line)]
        if lines:
            found.append(f"{label} (lines {', '.join(map(str, lines[:8]))})")
    return found


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10, cwd=Path(__file__).parent
        )
        return out.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def baseline_correlation(df: pd.DataFrame, weights: pd.Series) -> float | None:
    baseline_path = Path(__file__).resolve().parent.parent / "strategies" / "baseline_flow.py"
    if not baseline_path.exists():
        return None
    base_fn = load_module_fn(baseline_path)
    df_clean = enforce_schema(df)
    r_base = run_backtest(df_clean, coerce_weights(df_clean, base_fn(df_clean.copy())), fee_bps=5.0).hourly_returns
    r_strat = run_backtest(df_clean, weights, fee_bps=5.0).hourly_returns
    if np.std(r_base) < 1e-12 or np.std(r_strat) < 1e-12:
        return None
    return round(float(np.corrcoef(r_base, r_strat)[0, 1]), 2)


def run_record(res: dict, provenance: dict, strategy_name: str) -> dict:
    """One JSON line per run for the official log: what was run, on what, and what came out."""
    def sharpe(m: BacktestMetrics | None) -> list[float] | None:
        return None if m is None else [round(float(x), 4) for x in (m.sharpe, m.sharpe_ci_low, m.sharpe_ci_high)]

    h_eval = res.get("holdout_eval") or {}
    return {
        "utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "strategy": strategy_name,
        **provenance,
        "verdict": "PASS" if res["passed"] else "REJECT",
        "causality": {"passed": bool(res["causality"]["passed"]), "detail": res["causality"]["detail"]},
        "sharpe_5bps": sharpe(res.get("fee_curve", {}).get(5.0)),
        "holdout_status": h_eval.get("status"),
        "holdout_detail": h_eval.get("detail"),
        "holdout_sharpe": sharpe(res.get("holdout")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Syndicate look-ahead and cost-drag audit on a strategy file.")
    parser.add_argument("strategy", help="Path to strategy .py file defining compute_weights(df)")
    parser.add_argument("--data", default="data/train_1h.parquet", help="Path to input parquet file")
    parser.add_argument("--holdout", default="", help="Optional path to private holdout parquet file")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random causality cuts (default: fresh per run)")
    parser.add_argument("--n-random-cuts", type=int, default=8, help="Number of random causality cuts (default: 8)")
    parser.add_argument("--container", default="", help="Docker image for the strategy process (no network, no data files)")
    parser.add_argument("--log", default="", help="Append a JSON record of this run to this file")
    args = parser.parse_args()
    if args.container:
        # A missing image must stop the run here, not show up as a crashed strategy.
        probe = subprocess.run(["docker", "image", "inspect", args.container], capture_output=True) if shutil.which("docker") else None
        if probe is None or probe.returncode != 0:
            print(f"Error: Docker image {args.container} not available. Build it with `docker build -t {args.container} .`", file=sys.stderr)
            return 2
        os.environ[CONTAINER_ENV] = args.container

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: dataset not found at {data_path}. Run `python -m data.build_dataset` first.", file=sys.stderr)
        return 2
    holdout_path = Path(args.holdout) if args.holdout else None
    if holdout_path is not None and not holdout_path.exists():
        print(f"Error: holdout file not found at {holdout_path}.", file=sys.stderr)
        return 2
    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        print(f"Error: strategy file not found at {strategy_path}.", file=sys.stderr)
        return 2

    df = pd.read_parquet(data_path)
    holdout_df = pd.read_parquet(holdout_path) if holdout_path is not None else None
    if holdout_df is not None and (problem := holdout_start_problem(df, holdout_df)):
        # A data problem stops the run; it must not show up as a verdict on the strategy.
        print(f"Error: {problem}.", file=sys.stderr)
        return 2
    seed = args.seed if args.seed is not None else new_causality_seed()
    res = run_full_audit(df, strategy_path, holdout_df=holdout_df, causality_seed=seed, n_random_cuts=args.n_random_cuts)

    if strategy_path.resolve().name != "baseline_flow.py" and "weights" in res:
        res["baseline_corr"] = baseline_correlation(df, res["weights"])

    res["review_flags"] = review_flags(strategy_path)

    provenance = {
        "audit_version": AUDIT_VERSION,
        "commit": _git_commit(),
        "strategy_sha256": _sha256(strategy_path),
        "data_sha256": _sha256(data_path),
        "holdout_sha256": _sha256(holdout_path) if holdout_path is not None else None,
        "causality_seed": seed,
        "n_random_cuts": args.n_random_cuts,
        "isolation": f"container {args.container}" if args.container else "host process",
    }
    name = safe_text(strategy_path.name)
    # The log record is written first, so a report that fails to print cannot cost the run its record.
    if args.log:
        with open(args.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(run_record(res, provenance, name)) + "\n")
    if hasattr(sys.stdout, "reconfigure"):
        # A console or file encoding such as cp1251 must not stop the report from printing.
        sys.stdout.reconfigure(errors="backslashreplace")
    print(format_audit_report(res, name, provenance))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
