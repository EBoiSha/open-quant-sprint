# Changelog

## v1.7 (2026-11-02)

Rules and documentation (audit unchanged, version 1.3.1)
- README: a short section on why the repository is public and what participants get out of it.
- `RULES.md`: pushed work, forks and draft PRs included, is public right away.
- `RULES.md`: holes in the audit are reported through GitHub's private vulnerability reporting.

## v1.6 (2026-11-02)

Audit (audit version 1.3.1)
- The CLI stops with an error when the `--holdout` file does not start one hour after the `--data`
  panel ends. Before, a gap went through silently: rolling features at the start of the period ran on
  stale bars, and the report still said "bar-by-bar evaluation". An overlap now stops the run too,
  instead of being reported as a failed strategy.

Rules and documentation
- `RULES.md` lists the commands for the Nov 22 and Dec 7 runs, including how to build the panel that
  leads into the forward period.
- README quick start: a virtual environment, realistic timings, and how to open a PR from a fork.
- `CODEOWNERS`: strategies and contrib files now need the maintainer's approval too. Before, a PR that
  touched only those files could be approved by any collaborator, including another participant.

## v1.5 (2026-11-01)

Rules and documentation (audit unchanged, version 1.3.0)
- `RULES.md`: Docker is no longer a condition of the official run. The guard is the maintainer's read
  of every strategy file, the forward run with the network off, the freeze record and a published run
  log. A strategy that reads files or opens connections is listed as failed and not run on the forward
  period. `--container` stays as an optional, stricter setup.
- `LIMITATIONS.md` says plainly that without Docker the read is the only protection against file and
  network access, and that it is one person reading.

## v1.4 (2026-11-01)

Audit (audit version 1.3.0)
- `--container IMAGE` runs every strategy process in Docker with no network, a read-only filesystem,
  no capabilities and only the strategy file mounted. `Dockerfile` and `.dockerignore` build the image
  from `src/` and `requirements.lock`, without data.
- `--log FILE` appends a JSON record of the run (commit, hashes, seed, isolation, verdict, Sharpe).
  The report prints how the strategy process was isolated.
- `python -m src.freeze write|check` records and verifies SHA-256 of strategies, contrib and harness
  files, the lock file and the Dockerfile in `benchmarks/freeze.txt`.
- CI job "Strategy sandbox isolation (Docker)" on pushes to `main` builds the image and checks that
  the probe `tests/fixtures/escape_probe.py` can read neither the data nor the network from it.

Rules
- `RULES.md`: the official runs use the freeze record, the container and the run log; what happens
  when a strategy fails in an official run.

## v1.3 (2026-11-01)

Audit (`src/sandbox.py`, audit version 1.2.0)
- Bar-by-bar evaluation of the holdout and forward periods sends the strategy process one bar at a
  time; the next bar is sent only after the weights for the current one come back. Up to v1.1 the
  whole period was sent at once, and a strategy could read later bars through the call stack while
  the leak check reported no leak.
- New canary `tests/fixtures/frame_peek.py` and a test that bar-by-bar results are unchanged for
  honest strategies and that the canary is rejected.

Rules and documentation
- `RULES.md` describes how the official run is isolated and states that the forward runner without
  network or data access is still in progress; until then the maintainer's reading is the guard.
- `LIMITATIONS.md` adds the call-stack vector.
- Pull request template asks for 100-200 words of empirical notes, as in the Issues, and no longer
  refers to a paper file that does not exist yet.

## v1.2 (2026-11-01)

Rules and repository (audit unchanged, version 1.1.0)
- `benchmarks/LEADERBOARD.md` removed. The sprint is research, not a competition: results are reported per strategy with intervals, never ranked.
- `RULES.md`: strategy PRs are merged after review with any audit verdict (a REJECT is merged by hand); co-authorship needs a merged PR, not a passing CI check; Dec 7 is a proposed date.
- CI on pushes to `main` reports the audit of merged strategies without failing the run. Pull requests still show a red check on REJECT.

## v1.1 (2026-10-31)

Audit (`src/audit.py`, audit version 1.1.0)
- The last full bar is always a causality cut. In v1.0 the last 148 hours were never compared, so a leak confined to them passed.
- Random cuts use a fresh seed on every run (was fixed seed 0, so the cut timestamps could be computed and avoided) and there are 8 of them instead of 3. The report prints the seed; `--seed` reproduces a run, `--n-random-cuts` sets the number of cuts.
- Two canary strategies that passed v1.0 (`tests/fixtures/tail_peek.py`, `tests/fixtures/known_cuts_peek.py`) are now tests that must be rejected.

Rules and documentation
- `RULES.md`: measured interval half-widths (+-6 holdout, +-9 forward; were stated as +-3 and +-5), what PASS means, the official verdict protocol (maintainer run on the frozen file, 60 random cuts, forward run without data or network access), results reported without a ranking.
- `LIMITATIONS.md` lists the known gaps of the audit.
- Pull request template no longer asks for a PASS: any verdict is acceptable.

Repository
- CI takes only contributed files under `strategies/` and `src/contrib/` from a pull request and everything else from `main` (v1.0 restored a fixed list and missed `src/__init__.py`, `tests/conftest.py` and `pyproject.toml`). Renamed strategy files are audited too.
- `CODEOWNERS` covers the whole repository except contributed strategies and contrib files.
- `requirements.lock` pins the dependency versions used by CI; Python 3.11+.

## v1.0 (2026-10-30)

Audit (`src/audit.py`, `src/sandbox.py`)
- Every strategy call from the CLI runs in a fresh Python process in an empty working directory.
- Causality check uses 8 prefix cuts (40, 65, 85, 95, 99 % and three random) instead of 3; prefixes run before the full panel.
- Holdout is evaluated bar by bar and compared with a one-shot run; a difference fails the audit.
- A crash on the holdout, or a missing `--holdout` file, fails the audit instead of being skipped.
- Null test replaced: 200 daily-block permutations, gate on the mean null Sharpe (was a single permutation, so the result depended on the seed). In-sample p-value reported for information.
- 95 % block-bootstrap intervals for every Sharpe ratio.
- Tiered cost row: BTC/ETH 5 bps, others 7 bps, 6 least liquid by trailing 30-day volume 10 bps.
- Report header shows audit version, commit and SHA-256 of the strategy and data files.
- Correlation with `baseline_flow.py` and manual review flags added to the report.

Data (`data/`)
- `train_1h.parquet` extended back to 2025-01-01 00:00 UTC: 14,871 hours, 297,420 rows. The previous 144 days are byte-identical to the earlier file; the holdout is unchanged.
- On the extended panel `baseline_flow.py` and `template.py` fail the audit. Neither was changed. The baseline weight fixture and the leaderboard were regenerated.
- Last holdout bar (2026-10-18 14:00 UTC) replaced with the final closed values from Binance; it had been downloaded before the bar closed.
- `build_dataset.py`: `--start`/`--end` for fixed windows, unclosed bars dropped, existing files are not overwritten without `--overwrite`, synthetic data goes to `data/synthetic/`.

Repository
- MIT `LICENSE`, `NOTICE.md` for market data, `RULES.md`, `src/contrib/` for activation-PR features.
- CI audits strategies changed in a pull request using the harness and data from `main`.
- `CODEOWNERS` covers tests, features, stats, reference strategies and benchmarks.
