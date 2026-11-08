# Sprint 01 Rules (v1.8)

Changes to these rules are listed in `CHANGELOG.md`.

## Timeline (UTC)

| Date | Event |
| --- | --- |
| Nov 2 | The repository opens. No invitation is needed: fork it and open PRs from your fork. Activation PRs (features, infrastructure, tests) can be merged. |
| Nov 2 - Nov 8 | Strategy PRs may be opened as drafts. They are not merged before Nov 9. |
| Nov 9 | Official start. Strategy PRs are merged after review once the write-up is in and the causality check passes. Any audit verdict is accepted; a REJECT shows a red CI check and the maintainer merges it by hand. |
| Nov 21, 23:59 | Code freeze: tag `sprint-01-freeze`, SHA-256 of every strategy file recorded in `benchmarks/freeze.txt`. |
| Nov 22 | Preliminary report on the holdout period (Sep 12 - Oct 18). |
| Dec 7 | Final report on the forward period (Nov 22 - Dec 6), computed on the frozen code. Proposed date; it moves if the cohort agrees. |

Everything pushed to this repository or to a fork of it, draft PRs included, is public from the moment
it is pushed. Work locally if you do not want an idea seen before you open the PR; the PR has to be in
before the code freeze.

## Strategy contract

- `strategies/<name>.py` defines `compute_weights(df: pd.DataFrame) -> pd.Series` aligned with `df.index`.
- The weight at a row may depend only on rows with the same or earlier `timestamp`.
- The function uses only the `df` it receives. No file, network or environment access, no imports of
  `src.audit`, `src.backtest` or `src.sandbox`, no state carried between calls.
- Output must be deterministic for identical input.
- Each audit call runs in a separate Python process started in an empty temporary directory.
  The audit report lists code patterns that the maintainer reviews by hand (block [13]); every strategy
  is read before merge.

## Audit verdict

`python -m src.audit strategies/<name>.py` prints `[PASS] SURVIVED AUDIT` when all of the following hold:

1. Causality: weights on prefixes of the panel (the last full bar, 40, 65, 85, 95, 99 % and eight random
   cuts) match the full-panel weights within 1e-9; two runs on the same input are identical; weights are
   not all zero. The random cuts use a fresh seed on every run; the report prints it, and
   `--seed <n>` reproduces that run.
2. Data quality score >= 80.
3. Sharpe at 5 bps on the full panel > 0 and on the purged 30 % out-of-sample slice > 0.
4. Lag test: Sharpe with one extra hour of delay > 0.
5. Null pipeline check: mean Sharpe over 200 daily-block permutations of returns within +-0.75.
6. Wick check: at most 20 % of positive gross PnL comes from bars right after liquidation wicks.
7. When a holdout or forward file is passed, the strategy runs bar by bar on it without errors and the
   bar-by-bar weights equal the one-shot weights.

Sharpe ratios are reported with 95 % block-bootstrap intervals. Measured half-widths: about +-1.5
Sharpe points on the 20-month training panel, about +-6 on the 36-day holdout, about +-9 on the 14-day
forward period. A positive Sharpe on these samples does not establish an edge.

What PASS means: the strategy is causal on the tested cuts, reproducible, and has a positive point
estimate after costs. It is a hygiene gate, not evidence of an edge. In a stress test, 7 % of random
combinations of standard causal features passed, and none of them had an interval above zero. REJECT
is an equally valid research result.

The audit catches common look-ahead mistakes. It is not secure against a participant who sets out to
fool it; the rules rely on good faith. Known gaps are listed in [`LIMITATIONS.md`](LIMITATIONS.md).

## Official verdict

CI is a convenience for contributors, not the referee. The official verdict on every strategy is the
maintainer's own run:

- on the files recorded in `benchmarks/freeze.txt`, published before Nov 22. `python -m src.freeze write`
  hashes (SHA-256) every strategy, contrib and harness file plus `requirements.lock` and notes the
  commit; `python -m src.freeze check` has to pass before each run;
- with the harness at tag `sprint-01-freeze` and the versions in `requirements.lock`;
- with `--n-random-cuts 60`, so a leak that slips past the few random cuts of one CI run still gets caught;
- bar by bar on the holdout and forward periods: the strategy process gets one bar at a time and sees
  the next one only after it has returned weights for the current one;
- after the maintainer has read every strategy file. This read is the actual guard against a strategy
  that reads data files or opens a connection, because the audit does not stop either (see
  `LIMITATIONS.md`). It looks for file access, sockets and HTTP clients, `inspect`, `gc`, `importlib`,
  `exec`/`eval`, and names assembled from strings;
- with the network off for the forward run: the forward file is downloaded and hashed first, then the
  machine stays offline until the run is done;
- with `--log`, which appends one JSON line per run (commit, file hashes, seed, isolation, verdict,
  Sharpe and interval). The log is published with the report, with the SHA-256 of the forward file;
- in the Docker sandbox (`--container oqs-sandbox`, built with `docker build -t oqs-sandbox .`) if the
  maintainer's machine has Docker. Each strategy process then runs with no network, a read-only
  filesystem and no data, and that is a real barrier instead of a read. It is optional, and the
  report states which isolation was used.

If the read finds file or network access in a strategy, it is listed as failed for that reason and is
not run on the forward period. If a strategy fails in an official run (an error, a timeout), the report
lists it as failed with the message, and the code is not changed and re-run. A run is repeated only if
the harness or the machine failed, and the log shows both runs.

### The two official runs, step by step

The audit takes a panel (`--data`) and the period that follows it (`--holdout`), and stops with an
error if the period does not start exactly one hour after the panel ends.

Nov 22, holdout: `--data data/train_1h.parquet --holdout data/holdout_1h.parquet`. The two files are
contiguous already.

Dec 7, forward:

1. After 00:00 UTC on Dec 7, build the panel up to the start of the period, and the period itself.
   The builder drops bars that have not closed yet.
   ```
   python -m data.build_dataset --start 2025-01-01T00:00 --end 2026-11-22T00:00 --holdout-frac 0 --out-dir data/forward/panel
   python -m data.build_dataset --start 2026-11-22T00:00 --end 2026-12-07T00:00 --holdout-frac 0 --out-dir data/forward/period
   ```
   The period file has to end at 23:00 on Dec 6 and pass the data quality check. Record its SHA-256.
2. Switch the network off and run `python -m src.freeze check`.
3. For each strategy, once its file has been read:
   ```
   python -m src.audit strategies/<name>.py --data data/forward/panel/panel_1h.parquet --holdout data/forward/period/panel_1h.parquet --n-random-cuts 60 --log data/forward/log.jsonl
   ```
   The causality and cost checks then run on the longer panel (Jan 2025 - Nov 21, 2026). The forward
   result is the run's record in `log.jsonl` (`verdict`, `holdout_status`, `holdout_sharpe`); the
   `Holdout` line of the report shows the same result.
4. Publish the reports, `log.jsonl` and the hashes.

## Holdout and forward period

- The holdout file (Sep 12 - Oct 18) is built from public Binance data. It is treated as a public
  validation period: results are reported with intervals, and the Sharpe sign is not a pass/fail gate.
- The forward period starts after the code freeze, so no strategy can have been fitted to it.
  It is the only sealed test. Results are reported with intervals, without a pass/fail threshold.
- The maintainer runs each final evaluation once. No intermediate runs on either period.
- Results are reported per strategy (verdict, Sharpe, interval) next to the reference strategies, not
  as a ranking: with intervals of +-6 to +-9 Sharpe points, an ordering would mostly reflect noise.

## Co-authorship of the working paper

Co-authors are contributors with at least one merged PR (any audit verdict), and a documented contribution
(feature, infrastructure, strategy or review). It does not depend on the sign of a Sharpe ratio.

The working paper is a short write-up of the sprint, not a journal article: how the audit works, one
section per track (about a page, written by its contributor from the PR write-up) and the results with
their intervals. It is posted as a preprint after the Dec 7 report, planned for January 2027, archived
with a DOI on Zenodo and also on SSRN. Each author approves the final text before it is posted.

## License

Code contributions are licensed under the MIT License (`LICENSE`). Market data under `data/` is not
covered by it (`NOTICE.md`).

## Reporting a hole in the audit

Report it privately, not in a public issue or PR: use
[Report a vulnerability](https://github.com/valiyevoktay-cmd/open-quant-sprint/security/advisories/new)
on the repository's Security tab. Only you and the maintainer see the report. Confirmed reports are
credited in `CHANGELOG.md`.
