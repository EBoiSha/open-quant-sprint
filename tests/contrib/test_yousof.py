from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audit import holdout_start_problem
from src.contrib.yousof import DQS_MIN, check_continuation
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
