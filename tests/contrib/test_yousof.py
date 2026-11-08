from __future__ import annotations

import io
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audit import holdout_start_problem
from src.contrib.yousof import (
    CHECKED_VERSIONS,
    DQS_MIN,
    AuditSummary,
    audit_table,
    check_continuation,
    main,
    parse_audit_report,
    split_reports,
)
from src.quality import audit_dataset

TRAIN = Path("data/train_1h.parquet")
HOUR = pd.Timedelta(hours=1)
NEW_HOURS = 15 * 24  # the length of the forward period
OFF_HOURS = "non-zero funding_rate outside 00:00/08:00/16:00 UTC"

Split = tuple[pd.DataFrame, pd.DataFrame]


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    if not TRAIN.exists():
        pytest.skip("training panel not available")
    return pd.read_parquet(TRAIN)


@pytest.fixture(scope="module")
def split(train: pd.DataFrame) -> Split:
    """The panel without its last 15 days, and those 15 days.

    This is the shape of the forward check. The frames are shared by
    the tests, so a test changes a copy, never the frames themselves.
    """
    cut = train["timestamp"].max() - (NEW_HOURS - 1) * HOUR
    prev = train.loc[train["timestamp"] < cut].reset_index(drop=True)
    new = train.loc[train["timestamp"] >= cut].reset_index(drop=True)
    return prev, new


def _errors(problems: list[str]) -> list[str]:
    return [line for line in problems if line.startswith("error:")]


def _warnings(problems: list[str]) -> list[str]:
    return [line for line in problems if line.startswith("warning:")]


def _sym(new: pd.DataFrame) -> str:
    return sorted(new["symbol"].unique())[3]


def _fmt(ts: pd.Timestamp) -> str:
    return f"{ts:%Y-%m-%d %H:%M}"


def _hour(df: pd.DataFrame, i: int) -> pd.Timestamp:
    """The i-th distinct timestamp, with its time zone."""
    return df["timestamp"].drop_duplicates().sort_values().iloc[i]


# --- clean continuations ---------------------------------------------


def test_train_split_in_half_is_clean(train: pd.DataFrame) -> None:
    cut = _hour(train, train["timestamp"].nunique() // 2)
    prev = train.loc[train["timestamp"] < cut]
    new = train.loc[train["timestamp"] >= cut]
    assert check_continuation(prev, new) == []


def test_last_15_days_continue_the_panel(split: Split) -> None:
    prev, new = split
    downloaded = new["timestamp"].max() + HOUR
    assert new["timestamp"].nunique() == NEW_HOURS
    assert check_continuation(prev, new) == []
    assert check_continuation(prev, new, as_of=downloaded) == []


@pytest.mark.parametrize(
    ("prev_unit", "new_unit"), [("ms", "ns"), ("ns", "ms")]
)
def test_timestamp_unit_does_not_matter(
    split: Split, prev_unit: str, new_unit: str
) -> None:
    # A fresh download can carry ns timestamps while the stored panel
    # has ms.
    prev, new = split
    prev = prev.assign(timestamp=prev["timestamp"].dt.as_unit(prev_unit))
    new = new.assign(timestamp=new["timestamp"].dt.as_unit(new_unit))
    assert check_continuation(prev, new) == []


def test_overlap_is_found_across_timestamp_units(split: Split) -> None:
    # The keys of a ms panel and a ns file are compared on one unit, so
    # an overlapping hour is not missed.
    prev, new = split
    last = prev.loc[prev["timestamp"] == prev["timestamp"].max()]
    overlapping = pd.concat([last, new], ignore_index=True)
    prev = prev.assign(timestamp=prev["timestamp"].dt.as_unit("ms"))
    new = overlapping.assign(
        timestamp=overlapping["timestamp"].dt.as_unit("ns")
    )
    errors = _errors(check_continuation(prev, new))
    already = f"{len(last)} (timestamp, symbol) keys of new are already"
    assert any(already in line for line in errors)


def test_timestamps_stored_as_numbers_are_named(split: Split) -> None:
    # Epoch milliseconds as int64 would be read as nanoseconds: 1970.
    prev, new = split
    millis = new["timestamp"].dt.as_unit("ms").astype("int64")
    assert check_continuation(prev, new.assign(timestamp=millis)) == [
        "error: timestamp in new is int64, not a time; read as "
        "nanoseconds since 1970 it would be wrong"
    ]


def test_column_order_does_not_matter(split: Split) -> None:
    prev, new = split
    assert check_continuation(prev, new[new.columns[::-1]]) == []


def test_inputs_are_not_modified(split: Split) -> None:
    prev, new = split
    prev_before, new_before = prev.copy(), new.copy()
    check_continuation(prev, new, as_of=new["timestamp"].max())
    pd.testing.assert_frame_equal(prev, prev_before)
    pd.testing.assert_frame_equal(new, new_before)


# --- the four breaks from the Issue ----------------------------------


def test_dropped_hour_is_reported(split: Split) -> None:
    prev, new = split
    sym, hour = _sym(new), _hour(new, 100)
    dropped = (new["symbol"] == sym) & (new["timestamp"] == hour)
    assert check_continuation(prev, new.loc[~dropped]) == [
        f"error: {sym}: no bar at {_fmt(hour)} UTC (1 missing)"
    ]


def test_duplicated_row_is_reported(split: Split) -> None:
    prev, new = split
    row = new.iloc[[500]]
    sym, hour = row["symbol"].iloc[0], row["timestamp"].iloc[0]
    doubled = pd.concat([new, row], ignore_index=True)
    assert check_continuation(prev, doubled) == [
        "error: 1 duplicated (timestamp, symbol) rows in new: "
        f"{sym} at {_fmt(hour)}"
    ]


def test_symbol_shifted_by_one_hour_is_reported(split: Split) -> None:
    prev, new = split
    sym = _sym(new)
    shifted = new.copy()
    mask = shifted["symbol"] == sym
    shifted.loc[mask, "timestamp"] = shifted.loc[mask, "timestamp"] + HOUR
    start, end = new["timestamp"].min(), new["timestamp"].max()

    problems = check_continuation(prev, shifted)
    assert _errors(problems) == [
        f"error: {sym}: new starts at {_fmt(start + HOUR)} UTC, expected "
        f"{_fmt(start)}, one hour after prev ends",
        f"error: {sym}: new ends at {_fmt(end + HOUR)} UTC, most symbols "
        f"end at {_fmt(end)}",
    ]
    # The shift also moves the symbol's funding off the settlement
    # hours.
    assert any(sym in w and OFF_HOURS in w for w in _warnings(problems))


def test_funding_at_0500_is_a_warning_not_an_error(split: Split) -> None:
    prev, new = split
    sym = _sym(new)
    at_five = (new["symbol"] == sym) & (new["timestamp"].dt.hour == 5)
    row = new.index[at_five][0]
    broken = new.copy()
    broken.loc[row, "funding_rate"] = 1e-4

    problems = check_continuation(prev, broken)
    assert len(problems) == 1
    assert problems[0].startswith(f"warning: {sym}: {OFF_HOURS}")
    assert _fmt(new.loc[row, "timestamp"]) in problems[0]


# --- further breaks --------------------------------------------------


def test_whole_missing_hour_is_one_line(split: Split) -> None:
    prev, new = split
    hour = _hour(new, 200)
    n_sym = new["symbol"].nunique()
    problems = check_continuation(prev, new.loc[new["timestamp"] != hour])
    assert problems == [
        f"error: all {n_sym} symbols: no bar at {_fmt(hour)} UTC "
        "(1 missing)"
    ]


def test_gap_at_the_seam_is_reported(split: Split) -> None:
    prev, new = split
    start = new["timestamp"].min()
    n_sym = new["symbol"].nunique()
    problems = check_continuation(prev, new.loc[new["timestamp"] != start])
    assert problems == [
        f"error: all {n_sym} symbols: new starts at {_fmt(start + HOUR)} "
        f"UTC, expected {_fmt(start)}, one hour after prev ends"
    ]


def test_overlap_with_prev_is_reported(split: Split) -> None:
    prev, new = split
    last = prev.loc[prev["timestamp"] == prev["timestamp"].max()]
    overlapping = pd.concat([last, new], ignore_index=True)
    errors = _errors(check_continuation(prev, overlapping))
    already = f"{len(last)} (timestamp, symbol) keys of new are already"
    assert any(already in line for line in errors)
    assert any("new starts at" in line for line in errors)


@pytest.mark.parametrize("n_cut_short", [1, 10])
def test_symbols_ending_early_are_reported(
    split: Split, n_cut_short: int
) -> None:
    # With 10 of 20 cut short, the complete half must not be the one
    # reported.
    prev, new = split
    syms = sorted(new["symbol"].unique())[:n_cut_short]
    end = new["timestamp"].max()
    cut_short = new["symbol"].isin(syms) & (new["timestamp"] == end)
    assert check_continuation(prev, new.loc[~cut_short]) == [
        f"error: {', '.join(syms)}: new ends at {_fmt(end - HOUR)} UTC, "
        f"most symbols end at {_fmt(end)}"
    ]


@pytest.mark.parametrize("edge", ["min", "max"])
def test_existing_checks_pass_a_symbol_short_at_either_end(
    split: Split, edge: str
) -> None:
    # Why start and end are checked per symbol: with one symbol
    # missing its first or its last bar, the file-level start check
    # and the data quality gate of the audit both pass the file.
    prev, new = split
    hour = new["timestamp"].agg(edge)
    cut = (new["symbol"] == _sym(new)) & (new["timestamp"] == hour)
    broken = new.loc[~cut]
    assert holdout_start_problem(prev, broken) is None
    assert audit_dataset(broken).dqs >= DQS_MIN
    assert len(_errors(check_continuation(prev, broken))) == 1


def test_missing_and_extra_symbols_are_reported(split: Split) -> None:
    prev, new = split
    sym = _sym(new)
    n_sym = new["symbol"].nunique()
    assert check_continuation(prev, new.loc[new["symbol"] != sym]) == [
        f"error: 1 of the {n_sym} symbols in prev are missing from new: "
        f"{sym}"
    ]
    extra = new.loc[new["symbol"] == sym].assign(symbol="NEWCOINUSDT")
    widened = pd.concat([new, extra], ignore_index=True)
    assert check_continuation(prev, widened) == [
        "error: new has 1 symbols that prev does not: NEWCOINUSDT"
    ]


def test_unclosed_bars_are_reported(split: Split) -> None:
    prev, new = split
    last_open = new["timestamp"].max()
    half = pd.Timedelta(minutes=30)
    n_sym = new["symbol"].nunique()
    assert check_continuation(prev, new, as_of=last_open + half) == [
        f"error: {n_sym} rows open after {_fmt(last_open - half)} UTC "
        "(as_of minus one hour), so they had not closed at download "
        f"time: bars at {_fmt(last_open)} UTC, {n_sym} symbols"
    ]
    # A bar that opened exactly one hour before the download has
    # closed; an as_of without a time zone is read as UTC.
    naive = (last_open + HOUR).tz_localize(None)
    assert check_continuation(prev, new, as_of=naive) == []
    # An as_of that is not a time must not pass as "all bars closed".
    assert check_continuation(prev, new, as_of=pd.NaT) == [
        "error: as_of is not a time, so unclosed bars were not checked"
    ]


def test_schema_problems_are_reported(split: Split) -> None:
    prev, new = split
    assert check_continuation(prev, new.drop(columns="funding_rate")) == [
        "error: new is missing columns ['funding_rate']"
    ]
    assert check_continuation(prev, new.assign(mark_price=1.0)) == [
        "error: new has columns that prev does not: ['mark_price']"
    ]
    broken = new.assign(trades=new["trades"].astype("float64"))
    broken.loc[10, "trades"] = np.nan
    problems = check_continuation(prev, broken)
    assert len(problems) == 1
    assert problems[0].startswith("error: new does not fit the schema")


@pytest.mark.parametrize("key", ["timestamp", "symbol"])
def test_rows_without_a_key_are_reported(split: Split, key: str) -> None:
    # The row is reported, and so is the bar that is now missing.
    prev, new = split
    row = 2005  # not in the first or the last hour
    sym, hour = new.loc[row, "symbol"], new.loc[row, "timestamp"]
    broken = new.copy()
    broken.loc[row, key] = None
    assert check_continuation(prev, broken) == [
        "error: 1 rows in new have no timestamp or no symbol",
        f"error: {sym}: no bar at {_fmt(hour)} UTC (1 missing)",
    ]


def test_missing_prices_are_reported(split: Split) -> None:
    prev, new = split
    broken = new.copy()
    broken.loc[[10, 11], "close"] = np.nan
    problems = check_continuation(prev, broken)
    assert "error: missing values in new: close=2" in problems


def test_silent_funding_download_is_a_warning(split: Split) -> None:
    # data/build_dataset.py writes funding_rate = 0 when a symbol's
    # funding download comes back empty.
    prev, new = split
    sym = _sym(new)
    broken = new.copy()
    broken.loc[broken["symbol"] == sym, "funding_rate"] = 0.0
    assert check_continuation(prev, broken) == [
        "warning: funding_rate is zero at every settlement hour for "
        f"{sym}; did the funding download fail?"
    ]


def test_price_jump_at_the_seam_is_a_warning(split: Split) -> None:
    # A redenominated contract (PEPE -> 1000PEPE) continues the keys
    # but not the prices.
    prev, new = split
    sym = _sym(new)
    broken = new.copy()
    prices = ["open", "high", "low", "close"]
    broken.loc[broken["symbol"] == sym, prices] *= 1000.0
    problems = check_continuation(prev, broken)
    assert len(problems) == 1
    assert problems[0].startswith(f"warning: {sym}: first open in new is ")


def test_errors_come_before_warnings(split: Split) -> None:
    prev, new = split
    broken = new.loc[new["timestamp"] != _hour(new, 50)].copy()
    broken.loc[broken["symbol"] == _sym(new), "funding_rate"] = 0.0
    problems = check_continuation(prev, broken)
    kinds = [line.split(":")[0] for line in problems]
    assert kinds == ["error", "warning"]


# --- Part 2: the audit report as a table -----------------------------

REPORTS = Path(__file__).parent / "audit_reports"
PASS_REPORT = "baseline_flow_holdout_v1.3.1.txt"
REJECT_REPORT = "baseline_flow_v1.3.1.txt"
CAUSAL_14 = (
    "Causal invariance verified across 14 prefix cuts "
    "(max diff=0.00e+00, seed=20261107)"
)
WITHHELD_NOTE = (
    "not read, as they come below a strategy's error message, which "
    "before audit 1.3.2 could add lines of its own: lag test, null check, "
    "wick share, fee table, 5 bps row, out-of-sample Sharpe"
)
VERSION_NOTE = (
    "audit v{} is not one this parser was checked against (1.0.0 to "
    "1.3.2); compare the table with the report"
)

# What the saved reports hold, read off them by hand. Fields not named
# are None, problems [].
BASELINE = dict(
    strategy="baseline_flow.py", version="1.3.1", commit="f9c2892",
    verdict="REJECT", causality_passed=True, causality_detail=CAUSAL_14,
    dqs=99.93, sharpe_5bps=-1.53, sharpe_5bps_ci=(-3.00, 0.07),
    turnover_5bps=0.0522, oos_sharpe=2.17, oos_sharpe_ci=(-0.52, 4.86),
    lag1_sharpe=-1.53, lag2_sharpe=-1.53, lag_passed=True,
    null_mean=-0.0, null_sd=0.74, null_passed=True,
    wick_share_pct=2.2, wick_passed=True,
)
LEAKY = dict(
    strategy="leaky_example.py", version="1.3.1", commit="f9c2892",
    verdict="REJECT", causality_passed=False,
    causality_detail="Look-ahead leak detected at 2025-07-13 "
    "13:00:00+00:00: max|w_trunc - w_full| = 3.644e+00 > 1.0e-09 "
    "(seed=20261107)",
    dqs=99.93, sharpe_5bps=-9.21, sharpe_5bps_ci=(-10.85, -7.56),
    turnover_5bps=1.3648, oos_sharpe=-13.46,
    oos_sharpe_ci=(-16.52, -10.88), lag1_sharpe=-9.21,
    lag2_sharpe=-10.33, lag_passed=False, null_mean=-0.11,
    null_sd=0.83, null_passed=True, wick_share_pct=0.9,
    wick_passed=True,
)
EXPECTED: dict[str, dict] = {
    REJECT_REPORT: BASELINE,
    # Same strategy and data as above: the format changed between the
    # versions, the numbers read from it did not.
    "baseline_flow_v1.0.0.txt": {
        **BASELINE, "version": "1.0.0", "commit": "a8133ad",
        "causality_detail": "Causal invariance verified across 8 prefix "
        "cuts (max diff=0.00e+00)",
    },
    "leaky_example_v1.3.1.txt": LEAKY,
    # The same run with the official holdout window: one more line.
    "leaky_example_holdout_v1.3.1.txt": {
        **LEAKY, "holdout_passed": False,
        "holdout_detail": "Holdout look-ahead: one-shot weights differ "
        "from bar-by-bar weights (max diff=9.912e-02)",
    },
    # Every check passes but data quality.
    "baseline_flow_low_dqs_v1.3.1.txt": dict(
        strategy="baseline_flow.py", version="1.3.1", commit="f9c2892",
        verdict="REJECT", causality_passed=True,
        causality_detail="Causal invariance verified across 13 prefix "
        "cuts (max diff=0.00e+00, seed=20261107)",
        dqs=74.92, sharpe_5bps=2.85, sharpe_5bps_ci=(-0.74, 6.48),
        turnover_5bps=0.0508, oos_sharpe=4.27, oos_sharpe_ci=(-2.30, 10.90),
        lag1_sharpe=2.85, lag2_sharpe=2.69, lag_passed=True,
        null_mean=0.28, null_sd=1.71, null_passed=True,
        wick_share_pct=2.4, wick_passed=True,
    ),
    # Version 1.3.2 changed how error messages are printed, not the
    # numbers.
    "baseline_flow_v1.3.2.txt": {
        **BASELINE, "version": "1.3.2", "commit": "f339d91",
    },
    # An error message with report lines in it. Before 1.3.2 they were
    # lines of the report, a second FINAL VERDICT included, so nothing
    # below the message is read but the last verdict line.
    "forging_example_v1.3.1.txt": dict(
        strategy="forging_example.py", version="1.3.1", commit="f9c2892",
        verdict="REJECT", causality_passed=False,
        causality_detail="Strategy execution crashed: ValueError: bad "
        "window",
        dqs=99.93,
        problems=[WITHHELD_NOTE],
    ),
    # Since 1.3.2 the message is part of the causality line, and the
    # lines below it are the audit's own.
    "forging_example_v1.3.2.txt": dict(
        strategy="forging_example.py", version="1.3.2", commit="f339d91",
        verdict="REJECT", causality_passed=False,
        causality_detail="Strategy execution crashed: ValueError: bad "
        "window | Wick PnL Share: 0.1% (1 anomaly bars) [PASS - structural "
        "flow] | Null SR mean  : +0.01 (sd 0.50, hurdle |mean| < 0.75) "
        "[PASS] | 1-bar lag SR  : +3.00 | 2-bar lag SR: +2.90 | Retention: "
        "96.7% [PASS] | [3] RANK IC (1h fwd) : mean=+0.05000 | IC_IR=+0.500 "
        "| t=+9.0 ...[truncated]",
        dqs=99.93, sharpe_5bps=0.0, sharpe_5bps_ci=(math.nan, math.nan),
        turnover_5bps=0.0, oos_sharpe=0.0,
        oos_sharpe_ci=(math.nan, math.nan), lag1_sharpe=0.0,
        lag2_sharpe=0.0, lag_passed=False, null_mean=0.0, null_sd=0.0,
        null_passed=False, wick_share_pct=100.0, wick_passed=False,
    ),
    # The audit stops before the report (exit code 2).
    "holdout_late_start_error.txt": dict(
        problems=[
            "no audit report in the text, which starts with 'Error: "
            "holdout starts at 2026-09-12 16:00:00+00:00, expected "
            "2026-09-12 15:00:00+00:00, one hour after the --data panel "
            "ends.'"
        ],
    ),
    PASS_REPORT: dict(
        strategy="baseline_flow.py", version="1.3.1", commit="f9c2892",
        verdict="PASS", causality_passed=True, causality_detail=CAUSAL_14,
        dqs=99.95, sharpe_5bps=1.45, sharpe_5bps_ci=(-1.27, 4.15),
        turnover_5bps=0.0513, oos_sharpe=0.36, oos_sharpe_ci=(-4.66, 6.02),
        lag1_sharpe=1.45, lag2_sharpe=1.64, lag_passed=True,
        null_mean=0.50, null_sd=1.33, null_passed=True,
        wick_share_pct=1.8, wick_passed=True, holdout_passed=True,
        holdout_sharpe=3.10, holdout_sharpe_ci=(-6.78, 13.09),
        holdout_detail="bar-by-bar evaluation",
    ),
    "crashing_example_v1.3.1.txt": dict(
        strategy="crashing_example.py", version="1.3.1", commit="f9c2892",
        verdict="REJECT", causality_passed=False,
        causality_detail="Strategy execution crashed: ValueError: feature "
        "window of 5000 bars is longer than the panel",
        dqs=99.93,
        # Before 1.3.2 an error message can go on in lines of its own,
        # so nothing below it is read.
        problems=[WITHHELD_NOTE],
    ),
    # Before audit 1.0.0: no version line, no intervals, another null
    # check.
    "baseline_flow_6ed3535.txt": dict(
        strategy="baseline_flow.py", verdict="PASS", causality_passed=True,
        causality_detail="Causal invariance verified across 3 prefix "
        "cuts (max diff=0.00e+00)",
        dqs=99.93, sharpe_5bps=3.15, turnover_5bps=0.0504, oos_sharpe=4.19,
        lag1_sharpe=3.15, lag2_sharpe=3.14, lag_passed=True,
        wick_share_pct=2.4, wick_passed=True,
        problems=[
            "missing: audit version (no 'audit v...' line); the report is "
            "older than audit 1.0.0 or its format changed",
            "missing: null check (no 'Null SR mean' line)",
            "missing: 95 % interval of the Sharpe at 5 bps (no CI column "
            "in the fee table)",
        ],
    ),
}


def _report(name: str) -> str:
    return (REPORTS / name).read_text(encoding="utf-8")


def _edit(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def _plain(values: dict) -> dict:
    """``values`` with NaN as "nan": NaN is not equal to itself."""

    def plain(value: object) -> object:
        if isinstance(value, tuple):
            return tuple(plain(v) for v in value)
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        return value

    return {key: plain(value) for key, value in values.items()}


def _parsed(text: str) -> dict:
    return _plain(parse_audit_report(text).as_dict())


def _table(text: str) -> str:
    return audit_table(parse_audit_report(text))


def _expected(name: str, **changes: object) -> dict:
    return _plain({**AuditSummary().as_dict(), **EXPECTED[name], **changes})


def _rows(table: str) -> dict[str, list[str]]:
    """Result and value of each table row, by label."""
    rows = {}
    for line in table.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        if len(cells) == 3 and cells[0] not in ("Check", "---"):
            rows[cells[0]] = cells[1:]
    return rows


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_saved_report_is_read_field_by_field(name: str) -> None:
    assert _parsed(_report(name)) == _expected(name)


def test_every_saved_report_is_tested_and_noted() -> None:
    # The README next to the reports names the 'audit v...' line each
    # one came from.
    assert sorted(p.name for p in REPORTS.glob("*.txt")) == sorted(EXPECTED)
    readme = (REPORTS / "README.md").read_text(encoding="utf-8")
    for name, fields in EXPECTED.items():
        version = fields.get("version")
        line = f"`audit v{version}`" if version else "none"
        assert f"| `{name}` | {line} |" in readme
    assert {f["version"] for f in EXPECTED.values() if "version" in f} <= set(
        CHECKED_VERSIONS
    )


def test_table_of_a_pass_with_holdout() -> None:
    table = _table(_report(PASS_REPORT))
    assert table.splitlines() == [
        "#### `baseline_flow.py`",
        "",
        "| Check | Result | Value |",
        "| --- | --- | --- |",
        "| Verdict | PASS | audit v1.3.1, commit f9c2892 |",
        f"| Causality | pass | `{CAUSAL_14}` |",
        "| Sharpe @ 5 bps | pass | +1.45, 95% CI [-1.27, +4.15] "
        "includes 0 |",
        "| Out-of-sample Sharpe | pass | +0.36, 95% CI [-4.66, +6.02] "
        "includes 0 |",
        "| Lag test | pass | 2-bar lag Sharpe +1.64, 1-bar +1.45 |",
        "| Null check | pass | mean Sharpe +0.50, sd 1.33 |",
        "| Wick share | pass | 1.8% of positive PnL |",
        "| Turnover @ 5 bps | pass | 0.0513 |",
        "| Holdout | pass | +3.10, 95% CI [-6.78, +13.09] includes 0, "
        "`bar-by-bar evaluation` |",
    ]


def test_table_of_a_reject_shows_the_failed_check() -> None:
    table = _table(_report(REJECT_REPORT))
    rows = _rows(table)
    assert rows["Verdict"] == ["**REJECT**", "audit v1.3.1, commit f9c2892"]
    assert rows["Sharpe @ 5 bps"] == [
        "**FAIL**",
        "-1.53 (needs > 0), 95% CI [-3.00, +0.07] includes 0",
    ]
    del rows["Verdict"], rows["Sharpe @ 5 bps"]
    assert [result for result, _ in rows.values()] == ["pass"] * 6
    assert "Holdout" not in rows
    assert "verdict is" not in table and "Parser notes" not in table


def test_table_of_a_leak_keeps_its_pipes_inside_the_cell() -> None:
    table = _table(_report("leaky_example_v1.3.1.txt"))
    for line in table.splitlines():
        if line.startswith("|"):
            assert len(re.findall(r"(?<!\\)\|", line)) == 4, line
    result, value = _rows(table)["Causality"]
    assert result == "**FAIL**"
    assert r"max\|w_trunc - w_full\|" in value
    assert _rows(table)["Lag test"][0] == "**FAIL**"


def test_table_of_a_failed_holdout() -> None:
    table = _table(_report("leaky_example_holdout_v1.3.1.txt"))
    assert _rows(table)["Holdout"] == [
        "**FAIL**",
        "`Holdout look-ahead: one-shot weights differ from bar-by-bar "
        "weights (max diff=9.912e-02)`",
    ]


def test_table_of_an_audit_that_stopped_before_its_report() -> None:
    # What CI saves when the audit exits with code 2.
    name = "holdout_late_start_error.txt"
    problem = EXPECTED[name]["problems"][0]
    table = audit_table(parse_audit_report(_report(name)), "x")
    assert table.splitlines() == ["#### `x`", "", f"- `{problem}`"]


def test_error_message_cannot_pass_for_report_lines() -> None:
    # Before 1.3.2: the forged lines are not read, the verdict is the
    # audit's own and the note says what was left out.
    table = _table(_report("forging_example_v1.3.1.txt"))
    rows = _rows(table)
    assert rows.pop("Verdict")[0] == "**REJECT**"
    assert rows.pop("Causality")[0] == "**FAIL**"
    assert set(map(tuple, rows.values())) == {("?", "not read, see below")}
    assert f"- `{WITHHELD_NOTE}`" in table
    assert "+3.00" not in table and "Holdout" not in rows
    # Since 1.3.2: the same text inside the causality line is not taken
    # for the lines it imitates.
    rows = _rows(_table(_report("forging_example_v1.3.2.txt")))
    assert rows["Wick share"] == ["**FAIL**", "100.0% of positive PnL"]
    assert rows["Lag test"][1] == "2-bar lag Sharpe +0.00, 1-bar +0.00"


def test_error_message_in_the_holdout_line_before_1_3_2() -> None:
    # Lines above the holdout line are the audit's; below it, only the
    # last verdict line is read.
    holdout = (
        "    Holdout       : Sharpe=+3.10 [-6.78, +13.09] (includes 0) | "
        "Ann.Ret=+64.98% | MaxDD=-3.72% | bar-by-bar evaluation"
    )
    crashed = (
        "    Holdout       : [FAIL] Strategy crashed on holdout: "
        "ValueError: x\n    Wick PnL Share: 0.1% (1 anomaly bars) [PASS]"
    )
    text = _edit(_report(PASS_REPORT), holdout, crashed)
    summary = parse_audit_report(text)
    assert summary.holdout_passed is False
    assert summary.holdout_detail == (
        "Strategy crashed on holdout: ValueError: x"
    )
    assert (summary.sharpe_5bps, summary.oos_sharpe) == (1.45, 0.36)
    assert summary.wick_share_pct is None and summary.lag_passed is None
    assert summary.verdict == "PASS"
    assert summary.problems == [
        "not read, as they come below a strategy's error message, which "
        "before audit 1.3.2 could add lines of its own: lag test, null "
        "check, wick share"
    ]


def test_table_of_a_crash() -> None:
    # The values of a crashed run are printed as +0.00 and 0.0000,
    # which rounding cannot tell from slightly above the gate.
    report = _report("forging_example_v1.3.2.txt")
    rows = _rows(_table(report))
    assert rows["Sharpe @ 5 bps"] == ["?", "+0.00 (needs > 0), 95% CI n/a"]
    assert rows["Turnover @ 5 bps"] == ["?", "0.0000 (needs > 1e-05)"]
    assert rows["Causality"][0] == "**FAIL**"


def test_table_of_a_report_older_than_the_version_line() -> None:
    table = _table(_report("baseline_flow_6ed3535.txt"))
    rows = _rows(table)
    assert rows["Verdict"] == ["PASS", "no audit version line"]
    assert rows["Null check"] == ["?", "not read, see below"]
    assert rows["Sharpe @ 5 bps"] == ["pass", "+3.15"]
    notes = table.split("Parser notes:\n\n", 1)[1].splitlines()
    assert notes == [
        f"- `{p}`" for p in EXPECTED["baseline_flow_6ed3535.txt"]["problems"]
    ]


# --- robustness: a format that changes, output that is cut ------------


def test_missing_line_costs_only_its_own_values() -> None:
    lines = _report(REJECT_REPORT).splitlines(keepends=True)
    text = "".join(ln for ln in lines if "Out-of-Sample" not in ln)
    assert _parsed(text) == _expected(
        REJECT_REPORT, oos_sharpe=None, oos_sharpe_ci=None,
        problems=["missing: out-of-sample Sharpe (no 'Out-of-Sample' line)"],
    )


def test_unparseable_values_are_named_with_their_line() -> None:
    wick = "Wick PnL Share: 2.2% (3223 anomaly bars) [PASS - structural"
    text = _edit(_report(REJECT_REPORT), wick, wick.replace("2.2%", "high"))
    assert _parsed(text) == _expected(
        REJECT_REPORT, wick_share_pct=None,
        problems=[
            "unparseable: wick share in 'Wick PnL Share: high (3223 "
            "anomaly bars) [PASS - structural flow]'"
        ],
    )
    # Values that fail on one line share one entry.
    lag = "1-bar lag SR  : -1.53 | 2-bar lag SR: -1.53 | Retention: 0.0%"
    reworded = "1-bar lag SR  : n/a, 2-bar: n/a, PASS"
    text = _edit(_report(REJECT_REPORT), lag + " [PASS]", reworded)
    problems = parse_audit_report(text).problems
    assert problems == [
        "unparseable: 1-bar lag Sharpe, 2-bar lag Sharpe, lag test result "
        f"in '{reworded}'"
    ]


@pytest.mark.parametrize("version", ["1.4.0", "2.0.0rc1"])
def test_unknown_version_is_read_and_named(version: str) -> None:
    text = _edit(_report(REJECT_REPORT), "audit v1.3.1", f"audit v{version}")
    note = VERSION_NOTE.format(version)
    assert _parsed(text) == _expected(
        REJECT_REPORT, version=version, problems=[note]
    )
    assert f"- `{note}`" in _table(text)


def test_fee_table_is_read_by_its_header() -> None:
    # Without the DSR column, the cells still go to the right names.
    header = "  DSR(N=10)"
    row = "-1.53  [-3.00, +0.07] (includes 0)      0.000"
    text = _edit(_report(REJECT_REPORT), header, "")
    text = _edit(text, row, row.removesuffix("      0.000"))
    assert _parsed(text) == _expected(REJECT_REPORT)
    # A cell without a column name is not given one.
    text = _edit(_report(REJECT_REPORT), row, row + "      0.111")
    summary = parse_audit_report(text)
    assert (summary.sharpe_5bps, summary.turnover_5bps) == (None, None)
    assert summary.problems == [
        "unparseable: 5 bps row, 7 values under 6 columns in '5.0 bps"
        "      -1.53  [-3.00, +0.07] (includes 0)      0.000      0.111"
        "   -24.74%   -68.56%    0.0522'"
    ]


@pytest.mark.parametrize("name", [PASS_REPORT, "leaky_example_v1.3.1.txt"])
def test_output_cut_anywhere_gives_no_wrong_value(name: str) -> None:
    # A report cut off at any character: every value is either the
    # right one or None, and a lost value is always named.
    text = _report(name)
    full = _parsed(text)
    named = ("the text ends inside a line", "missing:", "no audit report")
    for cut in range(len(text) + 1):
        parsed = _parsed(text[:cut])
        problems = parsed.pop("problems")
        lost = [
            k for k, v in parsed.items() if v is None and full[k] is not None
        ]
        if lost:
            assert any(p.startswith(named) for p in problems), (cut, lost)
        assert all(
            v is None or v == full[k] for k, v in parsed.items()
        ), (cut, text[:cut][-40:])


def test_output_cut_in_a_number_is_not_read_as_another_number() -> None:
    text = _report(REJECT_REPORT)
    cut = text[: text.index("2-bar lag SR: -1.53") + len("2-bar lag SR: -1.5")]
    summary = parse_audit_report(cut)
    assert summary.lag2_sharpe is None and summary.verdict is None
    assert summary.problems[0] == (
        "the text ends inside a line, which is not read: '1-bar lag SR  : "
        "-1.53 | 2-bar lag SR: -1.5'"
    )
    assert "missing: verdict (no 'FINAL VERDICT' line)" in summary.problems
    assert _rows(audit_table(summary))["Verdict"] == [
        "?", "audit v1.3.1, commit f9c2892"
    ]


def test_verdict_and_rows_that_disagree_get_a_note() -> None:
    reject, accept = "[REJECT] FAILED", "[PASS] SURVIVED"
    passed = _edit(_report(REJECT_REPORT), reject, accept)
    assert (
        "The verdict is PASS, but Sharpe @ 5 bps failed here: the table and "
        "the audit disagree, so read the report."
    ) in _table(passed)
    rejected = _edit(_report(PASS_REPORT), accept, reject)
    assert (
        "The verdict is REJECT, but no row failed: the audit gates on "
        "something this table does not show, so read the report."
    ) in _table(rejected)


def test_data_quality_gets_a_row_only_below_the_gate() -> None:
    assert "Data quality" not in _rows(_table(_report(PASS_REPORT)))
    # A REJECT for the data alone: the row says why, so no note.
    table = _table(_report("baseline_flow_low_dqs_v1.3.1.txt"))
    rows = _rows(table)
    assert rows.pop("Data quality") == [
        "**FAIL**", f"DQS 74.92/100 (needs >= {DQS_MIN:g})"
    ]
    assert rows.pop("Verdict")[0] == "**REJECT**"
    assert {result for result, _ in rows.values()} == {"pass"}
    assert "The verdict is" not in table


@pytest.mark.parametrize(
    ("printed", "result"), [("-0.00", "**FAIL**"), ("+0.00", "?"),
                            ("+0.01", "pass")]
)
def test_rounding_decides_only_where_it_can(printed: str, result: str) -> None:
    # -0.00 is below zero; +0.00 is zero or just above it.
    row = "      -1.53  [-3.00, +0.07]"
    text = _edit(_report(REJECT_REPORT), row, row.replace("-1.53", printed))
    rows = _rows(_table(text))
    assert rows["Sharpe @ 5 bps"][0] == result


def test_free_text_cannot_add_markup_to_the_summary() -> None:
    message = "feature window of 5000 bars is longer than the panel"
    hostile = "see [docs](https://example.com) | `rm -rf` <b>now</b>"
    report = _report("crashing_example_v1.3.1.txt")
    table = _table(_edit(report, message, hostile))
    assert _rows(table)["Causality"][1] == (
        "`Strategy execution crashed: ValueError: see "
        r"[docs](https://example.com) \| 'rm -rf' <b>now</b>`"
    )
    long = _edit(report, message, "x" * 500)
    value = _rows(_table(long))["Causality"][1]
    assert len(value) == 162 and value.endswith("...`")


def test_several_reports_in_one_text() -> None:
    first, second = _report(REJECT_REPORT), _report(PASS_REPORT)
    both = first + second
    assert split_reports(both) == [first.rstrip("\n"), second.rstrip("\n")]
    summary = parse_audit_report(both)
    assert summary.verdict == "REJECT"
    assert summary.problems == [
        "the text holds 2 reports; only the first is read"
    ]


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("", "no audit report: the text is empty"),
        ("Error: dataset not found at data/x.parquet.\n",
         "no audit report in the text, which starts with 'Error: dataset "
         "not found at data/x.parquet.'"),
    ],
)
def test_text_without_a_report(text: str, problem: str) -> None:
    summary = parse_audit_report(text)
    assert summary.is_empty() and summary.problems == [problem]
    assert audit_table(summary, "x").splitlines() == [
        "#### `x`", "", f"- `{problem}`"
    ]


def test_command_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = [REPORTS / REJECT_REPORT, REPORTS / PASS_REPORT]
    tables = [_table(_report(p.name)) for p in reports]
    assert main([str(p) for p in reports]) == 0
    assert capsys.readouterr().out == "\n\n".join(tables) + "\n"
    # Several reports on stdin, as from a concatenated log.
    text = "".join(_report(p.name) for p in reports)
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    assert main([]) == 0
    assert capsys.readouterr().out == "\n\n".join(tables) + "\n"
    # An unreadable file is named and fails the command, not the rest.
    missing = tmp_path / "nothing.txt"
    assert main([str(missing), str(reports[0])]) == 1
    out = capsys.readouterr().out
    assert out.startswith(f"Could not read `{missing}`: ")
    assert out.endswith(tables[0] + "\n")


def test_runs_as_a_copied_script(tmp_path: Path) -> None:
    # The way CI calls it: a copy of the file as it is on main.
    repo = Path(__file__).resolve().parents[2]
    script = tmp_path / "audit_table.py"
    script.write_bytes((repo / "src/contrib/yousof.py").read_bytes())
    path = REPORTS / PASS_REPORT
    done = subprocess.run(
        [sys.executable, str(script), str(path)],
        capture_output=True, text=True, timeout=60, check=False,
        env={**os.environ, "PYTHONPATH": str(repo)},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == _table(_report(path.name)) + "\n"


def test_runs_as_a_module() -> None:
    path = REPORTS / PASS_REPORT
    done = subprocess.run(
        [sys.executable, "-m", "src.contrib.yousof", str(path)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == _table(_report(path.name)) + "\n"
