"""Track 01 (data & CI) by Yousof Sagr Shaladi: checks on new data,
and the audit report as a table for reviewers.

Part 1: new data
----------------

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

Part 2: the audit report as a table
-----------------------------------

CI writes the text of ``python -m src.audit`` into the job summary.
``parse_audit_report`` reads the values a reviewer looks at first into
an ``AuditSummary``, and ``audit_table`` turns that into a Markdown
table; ``python -m src.contrib.yousof REPORT...`` does both for CI.
``src/audit.py`` is neither changed nor imported: the format is taken
from its print statements (``format_audit_report``) and from saved
reports in ``tests/contrib/audit_reports/``.

How a change of format shows. Each value is read from its own line,
found by its label, so a moved or reworded line costs only its own
values. What cannot be read is None and named in ``problems``, as
"missing" (no such line) or "unparseable" (the line is there, the
value is not); nothing raises. The fee table is read by its header,
so a column that is added, dropped or moved is not taken for another.
Output cut off inside a line could read as another number there (-1.5
for -1.53), so a last line without a line break is not read.
An ``audit v...`` line outside CHECKED_VERSIONS, or none, is named
too: the lines read here and the verdict rule are the same from 1.0.0
to 1.3.1 (git history of src/audit.py), a later version may differ.

The Result column. The report tags four checks itself: causality, lag
test, null check and wick share; their tags are shown as they are.
Sharpe at 5 bps, out-of-sample Sharpe and turnover carry no tag, so
their result is the verdict rule of ``run_full_audit`` (> 0, > 0,
> 1e-5) applied to the printed value. Where rounding hides the answer
(+0.00, 0.0000) the result is "?". When the verdict and the rows
disagree, PASS with a failed row or REJECT without one, a note under
the table says so: the audit then gates on something the table does
not show, or its format changed. Data quality gets a row only below
the audit's gate; on the official panel it is the same for everyone.

Free text from the report goes into code spans, without backticks,
line breaks or bare pipes. The causality detail can hold a strategy's
exception message, and it must not add links or markup to the summary.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


# --- Part 2: the audit report as a table -----------------------------

# Audit versions this parser was checked against (see the docstring).
CHECKED_VERSIONS = ("1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.3.1")
TURNOVER_MIN = 1e-5  # the audit's turnover gate at 5 bps
DETAIL_MAX = 160  # characters of free text shown in the table

_HEADER = "SYNDICATE AUDIT REPORT ::"
_NUM = r"[+-]?(?:\d+(?:\.\d+)?|nan|inf)"
_CI = rf"\[{_NUM}, {_NUM}\]"
_TAG = r"\[(PASS|FAIL)\b[^\]]*\]"

Interval = tuple[float, float]


@dataclass
class AuditSummary:
    """The values of one audit report that a reviewer reads first.

    Numbers are as printed, so rounded: Sharpe ratios to 0.01, the
    wick share (in percent) to 0.1, turnover to 0.0001. An interval
    printed as "n/a" is (nan, nan), as in ``src.backtest``. A value
    is None when the report does not have it, and ``problems`` says
    why; not named there are the holdout values, which only a run
    with ``--holdout`` has, the commit and the out-of-sample interval.
    """

    strategy: str | None = None
    version: str | None = None
    commit: str | None = None
    verdict: str | None = None  # "PASS" or "REJECT"
    causality_passed: bool | None = None
    causality_detail: str | None = None
    dqs: float | None = None
    sharpe_5bps: float | None = None
    sharpe_5bps_ci: Interval | None = None
    turnover_5bps: float | None = None
    oos_sharpe: float | None = None
    oos_sharpe_ci: Interval | None = None
    lag1_sharpe: float | None = None
    lag2_sharpe: float | None = None
    lag_passed: bool | None = None
    null_mean: float | None = None
    null_sd: float | None = None
    null_passed: bool | None = None
    wick_share_pct: float | None = None
    wick_passed: bool | None = None
    holdout_passed: bool | None = None
    holdout_sharpe: float | None = None
    holdout_sharpe_ci: Interval | None = None
    holdout_detail: str | None = None
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def is_empty(self) -> bool:
        """True when not a single value was read."""
        values = self.as_dict()
        del values["problems"]
        return all(value is None for value in values.values())


def split_reports(text: str) -> list[str]:
    """``text`` cut into one piece per audit report, in order.

    A report starts at its "SYNDICATE AUDIT REPORT ::" line, or at the
    rule of "=" above it; text before the first report stays with it.
    A text without such a line is one piece, an empty text none.
    """
    if not text.strip():
        return []
    lines = text.splitlines()
    starts = [
        i - 1 if i > 0 and set(lines[i - 1].strip()) == {"="} else i
        for i, line in enumerate(lines)
        if line.lstrip().startswith(_HEADER)
    ]
    cuts = [0] + starts[1:] + [len(lines)]
    return ["\n".join(lines[a:b]) for a, b in zip(cuts, cuts[1:])]


def parse_audit_report(text: str) -> AuditSummary:
    """Read one ``python -m src.audit`` report into an AuditSummary.

    Of a text with several reports only the first is read, and
    ``problems`` says so. Odd input never raises: what cannot be read
    is None and named in ``problems``.
    """
    summary = AuditSummary()
    reports = split_reports(text)
    if not reports:
        summary.problems.append("no audit report: the text is empty")
        return summary
    if len(reports) > 1:
        summary.problems.append(
            f"the text holds {len(reports)} reports; only the first is "
            "read"
        )
    report = reports[0]
    if len(reports) == 1 and not text.endswith("\n"):
        # Output cut off inside a line: the rest of it could read as
        # another number (-1.5 for -1.53).
        head, _, tail = report.rpartition("\n")
        if tail.strip() and set(tail.strip()) != {"="}:
            summary.problems.append(
                "the text ends inside a line, which is not read: "
                f"{tail.strip()[:DETAIL_MAX]!r}"
            )
            report = head
    lines = _Lines(report, summary.problems)
    _read_head(lines, summary)
    _read_checks(lines, summary)
    _read_fee_row(lines, summary)
    _read_split(lines, summary)
    if summary.is_empty():
        first = reports[0].strip().splitlines()[0][:DETAIL_MAX]
        summary.problems[:] = [
            f"no audit report in the text, which starts with {first!r}"
        ]
    return summary


class _Lines:
    """The lines of one report, and what could not be read from them."""

    def __init__(self, text: str, problems: list[str]) -> None:
        self.lines = text.splitlines()
        self.problems = problems

    def find(
        self, what: str, label: str, anchor: str, required: bool = True
    ) -> str | None:
        """The first line matching the regular expression ``anchor``.

        ``label`` is how the anchor reads in the report.
        """
        for line in self.lines:
            if re.search(anchor, line):
                return line
        if required:
            self.problems.append(f"missing: {what} (no {label!r} line)")
        return None

    def match(
        self, what: str, line: str | None, pattern: str
    ) -> re.Match[str] | None:
        if line is None:
            return None
        found = re.search(pattern, line)
        if found is None:
            self.unparseable(what, line)
        return found

    def number(
        self, what: str, line: str | None, pattern: str
    ) -> float | None:
        found = self.match(what, line, pattern)
        return None if found is None else float(found[1])

    def passed(self, what: str, line: str | None) -> bool | None:
        """The [PASS] or [FAIL] tag of ``line``."""
        found = self.match(what, line, _TAG)
        return None if found is None else found[1] == "PASS"

    def unparseable(self, what: str, line: str) -> None:
        # Values that fail on the same line share one entry.
        tail = f" in {line.strip()!r}"
        last = self.problems[-1] if self.problems else ""
        if last.startswith("unparseable: ") and last.endswith(tail):
            self.problems[-1] = f"{last[:-len(tail)]}, {what}{tail}"
        else:
            self.problems.append(f"unparseable: {what}{tail}")


def _interval(text: str) -> Interval | None:
    if text == "n/a":
        return (math.nan, math.nan)
    found = re.fullmatch(rf"\[({_NUM}), ({_NUM})\]", text)
    return None if found is None else (float(found[1]), float(found[2]))


def _read_head(lines: _Lines, s: AuditSummary) -> None:
    """Strategy name, audit version and commit, and the verdict."""
    line = lines.find("strategy name", _HEADER, re.escape(_HEADER))
    found = lines.match("strategy name", line, r"::\s*(\S.*?)\s*$")
    if found:
        s.strategy = found[1]

    line = lines.find("", "", r"^\s*audit v", required=False)
    found = lines.match("audit version", line, r"audit v(\d[0-9A-Za-z.+-]*)")
    if line is None:
        lines.problems.append(
            "missing: audit version (no 'audit v...' line); the report "
            "is older than audit 1.0.0 or its format changed"
        )
    elif found:
        s.version = found[1]
        if s.version not in CHECKED_VERSIONS:
            lines.problems.append(
                f"audit v{s.version} is not one this parser was checked "
                f"against ({CHECKED_VERSIONS[0]} to "
                f"{CHECKED_VERSIONS[-1]}); compare the table with the "
                "report"
            )
        commit = re.search(r"\bcommit ([0-9a-f]{7,40})\b", line)
        s.commit = commit[1] if commit else None

    line = lines.find("verdict", "FINAL VERDICT", r"^\s*FINAL VERDICT\s*:")
    found = lines.match("verdict", line, r"\[(PASS|REJECT)\]")
    if found:
        s.verdict = found[1]


def _read_checks(lines: _Lines, s: AuditSummary) -> None:
    """Data quality, causality, lag test, null check and wick share."""
    line = lines.find("data quality", "DATA QUALITY", r"DATA QUALITY\s*:")
    s.dqs = lines.number("data quality score", line, rf"DQS=({_NUM})/")

    what = "causality check"
    line = lines.find(what, "CAUSALITY CHECK", r"CAUSALITY CHECK\s*:")
    found = lines.match(what, line, r"\[(PASS|FAIL)\]\s*(.*?)\s*$")
    if found:
        s.causality_passed = found[1] == "PASS"
        s.causality_detail = found[2]

    line = lines.find("lag test", "1-bar lag SR", r"1-bar lag SR\s*:")
    s.lag1_sharpe = lines.number(
        "1-bar lag Sharpe", line, rf"1-bar lag SR\s*:\s*({_NUM})"
    )
    s.lag2_sharpe = lines.number(
        "2-bar lag Sharpe", line, rf"2-bar lag SR\s*:\s*({_NUM})"
    )
    s.lag_passed = lines.passed("lag test result", line)

    line = lines.find("null check", "Null SR mean", r"Null SR mean\s*:")
    s.null_mean = lines.number(
        "null mean Sharpe", line, rf"Null SR mean\s*:\s*({_NUM})"
    )
    s.null_sd = lines.number("null sd", line, rf"\(sd ({_NUM})")
    s.null_passed = lines.passed("null check result", line)

    what = "wick share"
    line = lines.find(what, "Wick PnL Share", r"Wick PnL Share\s*:")
    s.wick_share_pct = lines.number(
        what, line, rf"Wick PnL Share\s*:\s*({_NUM})%"
    )
    s.wick_passed = lines.passed("wick check result", line)


def _read_fee_row(lines: _Lines, s: AuditSummary) -> None:
    """Sharpe, its interval and turnover from the 5 bps row.

    The cells are matched to the column names of the table header, so
    a new, dropped or moved column cannot pass for another one.
    """
    header = lines.find("fee table", "Fee (bps)", r"^\s*Fee \(bps\)")
    row = lines.find("5 bps row", "5.0 bps", r"^\s*5\.0 bps\s")
    if header is None or row is None:
        return
    names = re.split(r"\s{2,}", header.strip())[1:]
    rest = re.sub(r"^\s*5\.0 bps", "", row)
    cell = rf"{_CI}(?: \(includes 0\))?|n/a|{_NUM}%?"
    cells = [found[0] for found in re.finditer(cell, rest)]
    if len(cells) != len(names):
        lines.problems.append(
            f"unparseable: 5 bps row, {len(cells)} values under "
            f"{len(names)} columns in {row.strip()!r}"
        )
        return
    cols = dict(zip(names, cells))

    sharpe = cols.get("Sharpe", "")
    if re.fullmatch(_NUM, sharpe):
        s.sharpe_5bps = float(sharpe)
    else:
        lines.unparseable("Sharpe at 5 bps", row)
    ci_names = [name for name in names if "CI" in name.split()]
    if not ci_names:
        lines.problems.append(
            "missing: 95 % interval of the Sharpe at 5 bps (no CI "
            "column in the fee table)"
        )
    else:
        ci = cols[ci_names[0]].removesuffix(" (includes 0)")
        s.sharpe_5bps_ci = _interval(ci)
        if s.sharpe_5bps_ci is None:
            lines.unparseable("95 % interval at 5 bps", row)
    turnover = cols.get("Turnover", "")
    if re.fullmatch(_NUM, turnover):
        s.turnover_5bps = float(turnover)
    else:
        lines.unparseable("turnover at 5 bps", row)


def _read_split(lines: _Lines, s: AuditSummary) -> None:
    """Out-of-sample Sharpe and, if there is one, the holdout line."""
    what = "out-of-sample Sharpe"
    line = lines.find(what, "Out-of-Sample", r"^\s*Out-of-Sample\s*:")
    sharpe = rf"Sharpe=({_NUM})(?:\s+({_CI}|n/a))?"
    found = lines.match(what, line, sharpe)
    if found:
        s.oos_sharpe = float(found[1])
        s.oos_sharpe_ci = _interval(found[2]) if found[2] else None

    anchor = r"^\s*Holdout(?: Vault)?\s*:"
    line = lines.find("", "", anchor, required=False)
    if line is None:
        return
    rest = line.split(":", 1)[1].strip()
    if rest.startswith("[FAIL]"):
        s.holdout_passed = False
        s.holdout_detail = rest.removeprefix("[FAIL]").strip()
        return
    found = lines.match("holdout Sharpe", line, sharpe)
    if found:
        s.holdout_passed = True
        s.holdout_sharpe = float(found[1])
        s.holdout_sharpe_ci = _interval(found[2]) if found[2] else None
        last = rest.rsplit("|", 1)[-1].strip()
        s.holdout_detail = last if "=" not in last else None


def audit_table(summary: AuditSummary, name: str | None = None) -> str:
    """The summary as Markdown: a heading, the table and any notes.

    ``name`` is the heading when the report has no strategy name.
    """
    s = summary
    out = [f"#### {_code(s.strategy or name or 'audit report')}", ""]
    if s.is_empty():
        return "\n".join(out + [f"- {_code(p)}" for p in s.problems])
    rows = _rows(s)
    if s.verdict is None:
        verdict = "?"
    else:
        verdict = s.verdict if s.verdict == "PASS" else f"**{s.verdict}**"
    if s.version is None:
        source = "no audit version line"
    else:
        source = f"audit v{s.version}"
        source += f", commit {s.commit}" if s.commit else ""
    out += [
        "| Check | Result | Value |",
        "| --- | --- | --- |",
        f"| Verdict | {verdict} | {source} |",
    ]
    results = {True: "pass", False: "**FAIL**", None: "?"}
    out += [
        f"| {label} | {results[passed]} | {_cell(value)} |"
        for label, passed, value in rows
    ]
    notes = _notes(s, rows)
    if notes:
        out += [""] + notes
    if s.problems:
        out += ["", "Parser notes:", ""]
        out += [f"- {_code(p)}" for p in s.problems]
    return "\n".join(out)


Row = tuple[str, bool | None, str]


def _rows(s: AuditSummary) -> list[Row]:
    """(label, result, value) for each check below the verdict."""
    unread = "not read, see below"
    oos_ok = _above(s.oos_sharpe, 0.0, 0.01)
    sharpe_ok = _above(s.sharpe_5bps, 0.0, 0.01)
    turnover_ok = _above(s.turnover_5bps, TURNOVER_MIN, 1e-4)

    def gate(ok: bool | None, rule: str) -> str:
        return "" if ok else f" (needs {rule})"

    sharpe = unread
    if s.sharpe_5bps is not None:
        sharpe = _sharpe(
            s.sharpe_5bps, s.sharpe_5bps_ci, gate(sharpe_ok, "> 0")
        )
    oos = unread
    if s.oos_sharpe is not None:
        oos = _sharpe(s.oos_sharpe, s.oos_sharpe_ci, gate(oos_ok, "> 0"))
    lag = unread
    if s.lag2_sharpe is not None:
        lag = f"2-bar lag Sharpe {s.lag2_sharpe:+.2f}"
        if s.lag1_sharpe is not None:
            lag += f", 1-bar {s.lag1_sharpe:+.2f}"
    null = unread
    if s.null_mean is not None:
        null = f"mean Sharpe {s.null_mean:+.2f}"
        null += f", sd {s.null_sd:.2f}" if s.null_sd is not None else ""
    wick = unread
    if s.wick_share_pct is not None:
        wick = f"{s.wick_share_pct:.1f} % of positive PnL"
    turnover = unread
    if s.turnover_5bps is not None:
        turnover = f"{s.turnover_5bps:.4f}"
        turnover += gate(turnover_ok, f"> {TURNOVER_MIN:g}")

    causality = _code(s.causality_detail or "") or unread
    rows: list[Row] = [
        ("Causality", s.causality_passed, causality),
        ("Sharpe @ 5 bps", sharpe_ok, sharpe),
        ("Out-of-sample Sharpe", oos_ok, oos),
        ("Lag test", s.lag_passed, lag),
        ("Null check", s.null_passed, null),
        ("Wick share", s.wick_passed, wick),
        ("Turnover @ 5 bps", turnover_ok, turnover),
    ]
    if s.holdout_passed is not None or s.holdout_detail is not None:
        holdout = _code(s.holdout_detail or "")
        if s.holdout_sharpe is not None:
            parts = [_sharpe(s.holdout_sharpe, s.holdout_sharpe_ci)]
            holdout = ", ".join(parts + ([holdout] if holdout else []))
        rows.append(("Holdout", s.holdout_passed, holdout or unread))
    if s.dqs is not None and s.dqs < DQS_MIN:
        dqs = f"DQS {s.dqs:g}/100 (needs >= {DQS_MIN:g})"
        rows.append(("Data quality", False, dqs))
    return rows


def _notes(s: AuditSummary, rows: list[Row]) -> list[str]:
    """Where the verdict and the rows of the table disagree."""
    failed = [label for label, passed, _ in rows if passed is False]
    unsure = [label for label, passed, _ in rows if passed is None]
    if s.verdict == "PASS" and failed:
        return [
            f"The verdict is PASS, but {', '.join(failed)} failed here: "
            "the table and the audit disagree, so read the report."
        ]
    if s.verdict == "REJECT" and not failed and not unsure:
        return [
            "The verdict is REJECT, but no row failed: the audit gates "
            "on something this table does not show, so read the report."
        ]
    return []


def _above(x: float | None, gate: float, step: float) -> bool | None:
    """Whether a value printed as ``x`` is above ``gate``.

    The value lies within half a ``step`` of ``x``, and one printed as
    "-0.00" is below zero. None where the rounding hides the answer.
    """
    if x is None or math.isnan(x):
        return None
    low, high = x - step / 2, x + step / 2
    if x == 0.0 and math.copysign(1.0, x) < 0:
        high = 0.0
    if low > gate:
        return True
    if high <= gate:
        return False
    return None


def _sharpe(x: float, ci: Interval | None, gate: str = "") -> str:
    text = f"{x:+.2f}{gate}"
    if ci is None:
        return text
    low, high = ci
    if math.isnan(low) or math.isnan(high):
        return f"{text}, 95 % CI n/a"
    text += f", 95 % CI [{low:+.2f}, {high:+.2f}]"
    return text + (" includes 0" if low <= 0.0 <= high else "")


def _code(text: str) -> str:
    """``text`` as a one-line Markdown code span, "" for no text."""
    flat = " ".join(text.replace("`", "'").split())
    if len(flat) > DETAIL_MAX:
        flat = flat[: DETAIL_MAX - 3] + "..."
    return f"`{flat}`" if flat else ""


def _cell(text: str) -> str:
    """``text`` for a table cell: a bare pipe would end the cell."""
    return text.replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    """``python -m src.contrib.yousof [REPORT ...]``.

    Prints a Markdown table for each audit report in the REPORT files,
    or on stdin without arguments, e.g. into ``$GITHUB_STEP_SUMMARY``.
    Returns 1 when a file cannot be read, else 0, whatever the verdict.
    """
    paths = sys.argv[1:] if argv is None else argv
    texts: list[tuple[str | None, str]] = []
    blocks: list[str] = []
    status = 0
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                texts.append((Path(path).stem, fh.read()))
        except OSError as exc:
            reason = _code(exc.strerror or type(exc).__name__)
            blocks.append(f"Could not read {_code(path)}: {reason}")
            status = 1
    if not paths:
        texts.append((None, sys.stdin.read()))
    for name, text in texts:
        for report in split_reports(text) or [text]:
            blocks.append(audit_table(parse_audit_report(report), name))
    print("\n\n".join(blocks))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
