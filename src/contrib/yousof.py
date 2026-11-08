"""Track 01 (data & CI) by Yousof Sagr Shaladi: checks on new data.

The holdout and forward files are downloaded after the panel they
extend. ``check_continuation`` checks that such a file is a clean
continuation of that panel, symbol by symbol.

From Issue #1: the same symbols, columns and dtypes as the panel;
every symbol starting exactly one hour after it ends in the panel,
with no overlapping keys and no gaps; with ``as_of``, only bars that
had closed at download time; and, as a warning, funding outside
00:00/08:00/16:00 UTC.

Why the start is checked per symbol. ``holdout_start_problem`` in
``src.audit`` compares the first timestamp of the whole file with the
last one of the whole panel. If one symbol starts an hour late and the
others on time, the first timestamp of the file is still right and
that check passes. Here each symbol is compared with its own last bar
in the panel. The audit keeps its file-level check; this one does not
replace it. ``src.audit`` is not imported: strategies may import
contrib modules but not the audit (README).

Checks beyond the Issue, and what each is for:

- Rows off the hourly grid. A bar at 10:30 is neither a duplicate nor
  a missing hour, so it would count as a bar of its own.
- A symbol that ends before or after the others. A download cut short
  for one symbol leaves no gap inside the file, only a shorter tail,
  and the last hours then lack that symbol in the cross-section.
- Funding that is zero at every settlement hour of a symbol, as a
  warning. ``data/build_dataset.py`` writes 0 when the funding
  download comes back empty, so a failed download reads as free
  funding.
- A first open more than 5 % from the symbol's last close in the
  panel, as a warning. A redenominated contract (PEPE -> 1000PEPE)
  keeps its place in the keys but not its prices, and the return
  across the seam is wrong by that factor. In the train panel an open
  is never more than 0.25 % from the previous close.
- Rows without a timestamp or a symbol. They have no place on the
  grid; they are reported and left out of the other checks.
- Timestamps stored as numbers. ``enforce_schema`` would read epoch
  milliseconds as nanoseconds, and every bar would land on 1970-01-01.
- Missing values, OHLC and volume sanity, non-positive prices and the
  data quality score, all from ``src.quality.audit_dataset``. The
  audit scores only the panel passed as ``--data``; the new file is
  passed as ``--holdout`` and is not scored there, so it is scored
  here, on its own.

A file with one of the first four passes the audit as it is: the
file-level start check finds no problem, and the data quality score,
even of the new file alone, stays above 99 (the gate is 80).
"""

from __future__ import annotations

import pandas as pd

from src.quality import audit_dataset
from src.schema import REQUIRED_COLS, enforce_schema

HOUR = pd.Timedelta(hours=1)
KEYS = ["timestamp", "symbol"]
FUNDING_HOURS = (0, 8, 16)
FUNDING_EPS = 1e-9  # same threshold as src/quality.py
DQS_MIN = 80.0  # the audit's data quality gate (src/audit.py)
SEAM_JUMP = 0.05  # first open vs. last close: a different contract?
SILENT_FUNDING_MIN = 3  # settlement hours before all-zero is flagged
MAX_EXAMPLES = 5


def check_continuation(
    prev: pd.DataFrame,
    new: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
) -> list[str]:
    """List what keeps ``new`` from cleanly continuing ``prev``.

    One readable line per problem. A line starts with "error:" when the
    file must not be evaluated as it is, and with "warning:" when it
    needs a look but can be legitimate (Binance can move a symbol to
    another funding interval). Errors come first. An empty list means
    OK.

    ``as_of`` is the download time; without a time zone it is read as
    UTC. With it, bars that had not closed by then are reported.
    """
    errors, frames = _prepare(prev, new)
    warnings: list[str] = []
    if frames is not None:
        prev, new = frames
        off_grid = new["timestamp"] != new["timestamp"].dt.floor("h")
        on_grid = new.loc[~off_grid]
        errors += _symbol_errors(prev, new)
        errors += _key_errors(prev, new, off_grid)
        errors += _seam_errors(prev, on_grid)
        warnings += _seam_jump_warnings(prev, on_grid)
        if as_of is not None:
            errors += _unclosed_errors(new, as_of)
        errors += _quality_errors(new)
        warnings += _funding_warnings(on_grid)
    lines = [f"error: {msg}" for msg in errors]
    lines += [f"warning: {msg}" for msg in warnings]
    return lines


def _fmt(ts: pd.Timestamp) -> str:
    return f"{ts:%Y-%m-%d %H:%M}"


def _hours(stamps: pd.Series | pd.DatetimeIndex) -> str:
    """Up to MAX_EXAMPLES distinct hours, oldest first, and a count."""
    stamps = pd.DatetimeIndex(stamps).unique().sort_values()
    shown = ", ".join(_fmt(ts) for ts in stamps[:MAX_EXAMPLES])
    more = len(stamps) - MAX_EXAMPLES
    return shown + (f" (+{more} more)" if more > 0 else "")


def _by_symbol(rows: pd.DataFrame) -> str:
    """The hours of ``rows``, for up to MAX_EXAMPLES symbols."""
    parts = [
        f"{sym} at {_hours(grp['timestamp'])}"
        for sym, grp in rows.groupby("symbol")
    ]
    shown = "; ".join(parts[:MAX_EXAMPLES])
    more = len(parts) - MAX_EXAMPLES
    return shown + (f"; +{more} more symbols" if more > 0 else "")


def _utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    """``ts`` in UTC; without a time zone it is read as UTC."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _prepare(
    prev: pd.DataFrame, new: pd.DataFrame
) -> tuple[list[str], tuple[pd.DataFrame, pd.DataFrame] | None]:
    """Column and schema errors, and both frames ready to compare.

    The frames went through ``enforce_schema``, have no rows without a
    key and carry ns timestamps. They are None when the two files
    cannot be compared at all.
    """
    named = (("prev", prev), ("new", new))
    errors = []
    # Columns first: enforce_schema raises on a missing one and
    # silently drops extra ones.
    for name, df in named:
        missing = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing:
            errors.append(f"{name} is missing columns {missing}")
    if errors:
        return errors, None
    extra = [col for col in new.columns if col not in prev.columns]
    if extra:
        errors.append(f"new has columns that prev does not: {extra}")
    lacking = [col for col in prev.columns if col not in new.columns]
    if lacking:
        errors.append(f"new lacks columns that prev has: {lacking}")

    frames = []
    for name, df in named:
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            # enforce_schema would read the numbers as nanoseconds
            # since 1970 and every bar would land on 1970-01-01.
            errors.append(
                f"timestamp in {name} is {df['timestamp'].dtype}, not a "
                "time; read as nanoseconds since 1970 it would be wrong"
            )
            continue
        try:
            frame = enforce_schema(df)
        except (ValueError, TypeError) as exc:
            errors.append(f"{name} does not fit the schema: {exc}")
            continue
        keyless = frame[KEYS].isna().any(axis=1)
        if keyless.any():
            errors.append(
                f"{int(keyless.sum())} rows in {name} have no timestamp "
                "or no symbol"
            )
        frame = frame.loc[~keyless]
        if frame.empty:
            errors.append(f"{name} has no rows to check")
            continue
        # enforce_schema keeps the timestamp unit: a fresh download can
        # carry ns where the stored panel has ms. That is not a
        # problem, so compare (and match keys) on one unit.
        stamps = frame["timestamp"].dt.as_unit("ns")
        frames.append(frame.assign(timestamp=stamps))
    if len(frames) < 2:
        return errors, None

    prev, new = frames
    # enforce_schema casts both frames to the schema, so this fires only
    # if enforce_schema itself stops doing that.
    differing = [
        f"{col} {new[col].dtype} vs {prev[col].dtype}"
        for col in REQUIRED_COLS
        if new[col].dtype != prev[col].dtype
    ]
    if differing:
        errors.append(
            "dtypes differ from prev after enforce_schema: "
            + ", ".join(differing)
        )
    return errors, (prev, new)


def _symbol_errors(prev: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    prev_syms, new_syms = set(prev["symbol"]), set(new["symbol"])
    errors = []
    missing = sorted(prev_syms - new_syms)
    if missing:
        errors.append(
            f"{len(missing)} of the {len(prev_syms)} symbols in prev are "
            f"missing from new: {', '.join(missing)}"
        )
    extra = sorted(new_syms - prev_syms)
    if extra:
        errors.append(
            f"new has {len(extra)} symbols that prev does not: "
            + ", ".join(extra)
        )
    return errors


def _key_errors(
    prev: pd.DataFrame, new: pd.DataFrame, off_grid: pd.Series
) -> list[str]:
    """Rows of ``new`` off the grid, duplicated or in ``prev``."""
    errors = []
    if off_grid.any():
        errors.append(
            f"{int(off_grid.sum())} rows in new are not on the hourly "
            f"grid: {_by_symbol(new.loc[off_grid])}"
        )
    dup = new.duplicated(KEYS, keep="first")
    if dup.any():
        errors.append(
            f"{int(dup.sum())} duplicated (timestamp, symbol) rows in "
            f"new: {_by_symbol(new.loc[dup])}"
        )
    new_keys = pd.MultiIndex.from_frame(new[KEYS])
    overlap = new_keys.isin(pd.MultiIndex.from_frame(prev[KEYS]))
    if overlap.any():
        errors.append(
            f"{int(overlap.sum())} (timestamp, symbol) keys of new are "
            f"already in prev: {_by_symbol(new.loc[overlap])}"
        )
    return errors


def _seam_errors(prev: pd.DataFrame, on_grid: pd.DataFrame) -> list[str]:
    """Per symbol: where ``new`` starts and ends, and missing hours.

    A symbol has to start one hour after it ends in ``prev``, end with
    the other symbols and have every hour in between. Symbols with the
    same problem share one line, so an hour missing across the panel is
    one line, not 20.
    """
    if on_grid.empty:
        return []
    prev_end = prev.groupby("symbol")["timestamp"].max()
    stamps = on_grid.drop_duplicates(KEYS).groupby("symbol")["timestamp"]
    starts, ends = stamps.min(), stamps.max()
    # On a tie the later end is the reference: a file cut short must
    # not make the complete symbols look wrong.
    common_end = ends.mode().max()
    checked = sorted(set(starts.index) & set(prev_end.index))
    found: dict[str, list[str]] = {}
    for sym in checked:
        start, end = starts[sym], ends[sym]
        expected = prev_end[sym] + HOUR
        if start != expected:
            msg = (
                f"new starts at {_fmt(start)} UTC, expected "
                f"{_fmt(expected)}, one hour after prev ends"
            )
            found.setdefault(msg, []).append(sym)
        if end != common_end:
            msg = (
                f"new ends at {_fmt(end)} UTC, most symbols end at "
                f"{_fmt(common_end)}"
            )
            found.setdefault(msg, []).append(sym)
        grid = pd.date_range(start, end, freq="h")
        gaps = grid.difference(pd.DatetimeIndex(stamps.get_group(sym)))
        if not gaps.empty:
            msg = f"no bar at {_hours(gaps)} UTC ({len(gaps)} missing)"
            found.setdefault(msg, []).append(sym)

    errors = []
    for msg, syms in found.items():
        everyone = len(syms) == len(checked) > 1
        who = f"all {len(syms)} symbols" if everyone else ", ".join(syms)
        errors.append(f"{who}: {msg}")
    return errors


def _seam_jump_warnings(
    prev: pd.DataFrame, on_grid: pd.DataFrame
) -> list[str]:
    """Symbols whose first open is far from their last close in prev.

    A redenominated contract (PEPE -> 1000PEPE) continues the keys but
    not the prices. Only symbols that start at the seam are compared.
    """
    # enforce_schema sorted both frames by timestamp.
    last = prev.drop_duplicates("symbol", keep="last")
    first = on_grid.drop_duplicates("symbol", keep="first")
    seam = first.merge(last, on="symbol", suffixes=("_new", "_prev"))
    seam = seam[seam["timestamp_new"] == seam["timestamp_prev"] + HOUR]
    seam = seam.sort_values("symbol")
    jump = (seam["open_new"] / seam["close_prev"] - 1.0).abs()
    return [
        f"{sym}: first open in new is {size:.1%} away from the last "
        "close in prev; same contract?"
        for sym, size in zip(seam["symbol"], jump)
        if size > SEAM_JUMP
    ]


def _unclosed_errors(new: pd.DataFrame, as_of: pd.Timestamp) -> list[str]:
    """Bars that had not closed at download time.

    A bar opening at t closes at t + 1h, so no bar may open later than
    ``as_of`` minus one hour.
    """
    cutoff = _utc(as_of) - HOUR
    if pd.isna(cutoff):
        return ["as_of is not a time, so unclosed bars were not checked"]
    late = new.loc[new["timestamp"] > cutoff]
    if late.empty:
        return []
    return [
        f"{len(late)} rows open after {_fmt(cutoff)} UTC (as_of minus "
        "one hour), so they had not closed at download time: bars at "
        f"{_hours(late['timestamp'])} UTC, "
        f"{late['symbol'].nunique()} symbols"
    ]


def _quality_errors(new: pd.DataFrame) -> list[str]:
    """Missing values and the single-panel checks of src/quality.py."""
    errors = []
    nulls = new[list(REQUIRED_COLS)].isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        counts = ", ".join(f"{col}={k}" for col, k in nulls.items())
        errors.append(f"missing values in new: {counts}")
    report = audit_dataset(new)
    if report.ohlc_violations:
        errors.append(
            f"{report.ohlc_violations} rows in new break OHLC or volume "
            "sanity (src/quality.py)"
        )
    if report.non_positive_prices:
        errors.append(
            f"{report.non_positive_prices} rows in new have non-positive "
            "prices"
        )
    if report.dqs < DQS_MIN:
        errors.append(
            f"data quality score of new is {report.dqs}, below the "
            f"audit's gate of {DQS_MIN:g}"
        )
    return errors


def _funding_warnings(on_grid: pd.DataFrame) -> list[str]:
    """Funding outside the settlement hours, or zero at all of them.

    Rows off the hourly grid are left out: they are an error already.
    """
    nonzero = on_grid["funding_rate"].abs() > FUNDING_EPS
    settle = on_grid["timestamp"].dt.hour.isin(FUNDING_HOURS)
    warnings = []
    for sym, grp in on_grid.loc[nonzero & ~settle].groupby("symbol"):
        warnings.append(
            f"{sym}: non-zero funding_rate outside 00:00/08:00/16:00 UTC "
            f"at {_hours(grp['timestamp'])}; Binance may have moved it to "
            "another funding interval"
        )
    # The builder writes 0 when a symbol's funding download comes back
    # empty.
    paid = nonzero[settle].groupby(on_grid.loc[settle, "symbol"])
    paid = paid.agg(["any", "size"])
    quiet = ~paid["any"] & (paid["size"] >= SILENT_FUNDING_MIN)
    silent = sorted(paid.index[quiet])
    if silent:
        warnings.append(
            "funding_rate is zero at every settlement hour for "
            f"{', '.join(silent)}; did the funding download fail?"
        )
    return warnings
