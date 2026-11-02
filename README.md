# open-quant-sprint

Backtest and look-ahead audit harness for cross-sectional strategies on 20 Binance USD-M perpetuals (1h bars). Used for Sprint 01 (Nov 9 - Dec 7, 2026).

Rules, timeline, verdict criteria: [`RULES.md`](RULES.md). Known gaps: [`LIMITATIONS.md`](LIMITATIONS.md). Changes: [`CHANGELOG.md`](CHANGELOG.md).

The audit catches common look-ahead mistakes; it is not secure against deliberate cheating, and a PASS is a hygiene check, not evidence of an edge (see [Audit](#audit)).

## Why it is public

Everything here is open: the harness, the data pipeline, every strategy and every audit report. The
idea is that each participant leaves the sprint with work anyone can check, not a screenshot of a
backtest.

- Your strategy and features stay yours: your own files, a PR under your name, and the audit
  scorecard in CI and in the final report.
- Anyone can clone the repository, rerun your audit and get the same numbers. That is the part worth
  linking on a CV or walking through in an interview.
- Contributors with a merged PR are co-authors of the working paper ([`RULES.md`](RULES.md#co-authorship-of-the-working-paper)).
- The code is MIT, so you can reuse what you wrote here anywhere, and so can everyone else.

Be straight about what a result means. A PASS says the strategy is causal, reproducible and positive
after costs on this panel; it does not prove an edge. A REJECT with a clear write-up of why the idea
failed counts just as much, and usually makes for a better conversation.

What you push is public right away: branches, forks and draft PRs alike (GitHub has no private forks
of a public repository). If you want to keep an idea to yourself for a while, work locally and open
the PR before the code freeze (Nov 21, 23:59 UTC).

## Quick start

Requires Python 3.11+. The lock file pins exact versions, so use a fresh virtual environment rather
than your global Python.

```bash
git clone https://github.com/valiyevoktay-cmd/open-quant-sprint
cd open-quant-sprint
python -m venv .venv                              # then .venv\Scripts\activate on Windows, source .venv/bin/activate elsewhere
pip install -r requirements.lock                  # exact versions used by CI, about 2 min
pip install -e . --no-deps
python -m pytest -q                               # expected: all tests pass, about 2 min
python -m src.audit strategies/baseline_flow.py   # expected: FINAL VERDICT: [REJECT], about 1 min
```

The baseline is expected to fail: see [Reference strategies](#reference-strategies).

## Repository layout

| Path | Contents |
| --- | --- |
| `data/train_1h.parquet` | Training panel, 297,420 rows |
| `data/build_dataset.py` | Binance REST download, cleaning, train/holdout split |
| `src/audit.py` | Audit CLI and verdict |
| `src/backtest.py` | Vectorized backtest: fills, fees, funding |
| `src/sandbox.py` | Runs each strategy call in a separate process, optionally in a Docker container (`--container`) |
| `src/freeze.py` | Code freeze record: SHA-256 of strategies and harness (`write`, `check`) |
| `Dockerfile` | Optional sandbox image for `--container`: harness and pinned dependencies, no data |
| `src/features.py` | Shared causal features |
| `src/contrib/<name>.py` | Activation-PR features, one file per contributor |
| `src/stats.py` | IC, Deflated Sharpe, bootstrap intervals |
| `src/quality.py`, `src/schema.py` | Data quality score, column schema |
| `strategies/` | Strategy files, one `compute_weights` each |
| `tests/` | Pytest suite and reference fixtures |

## Dataset

- 20 symbols, 1h bars, open times 2025-01-01 00:00 to 2026-09-12 14:00 UTC (14,871 hours).
- Columns: `timestamp` (bar open, UTC), `symbol`, `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `taker_buy_quote_volume`, `trades`, `funding_rate`.
- `funding_rate` is non-zero only at 00:00, 08:00, 16:00 UTC.
- The file in the repository is the reference dataset. Rebuild only to check it:
  `python -m data.build_dataset --start 2025-01-01T00:00 --end 2026-09-12T15:00 --holdout-frac 0 --out-dir data/rebuild`
- The holdout (Sep 12 - Oct 18, 2026) is not in the repository.
- Offline synthetic panel: `python -m data.build_dataset --synthetic` (written to `data/synthetic/`).

## Writing a strategy

`strategies/<name>.py`:

```python
import pandas as pd

def compute_weights(df: pd.DataFrame) -> pd.Series:
    # df: 11 columns above, sorted by (timestamp, symbol).
    # Return weights aligned with df.index. Gross exposure is scaled to sum(|w|) <= 1 per timestamp.
    ...
```

- The weight at a row may use only rows with the same or earlier `timestamp`.
- Use only the `df` argument: no files, network, environment, or state kept between calls.
- No imports of `src.audit`, `src.backtest`, `src.sandbox`.
- Same input, same output.
- Start from `strategies/template.py`. Shared features: `src/features.py`.

Workflow:

```bash
git checkout -b feat/<name>
python -m src.audit strategies/<name>.py
git remote add fork https://github.com/<you>/open-quant-sprint   # once, after forking on GitHub
git push fork feat/<name>       # open a PR from your fork; strategy PRs stay draft until Nov 9
```

No invitation or write access is needed: fork the repository on GitHub (the Fork button) and open
the PR from your fork. Until you have a merged PR, each CI run waits for the maintainer to approve it;
after that CI starts on its own.

CI runs the test suite and audits every strategy file added, changed or renamed in the PR. Only your files under `strategies/` and `src/contrib/` are taken from the PR; everything else comes from `main`. CI is a convenience: the official verdict is the maintainer's run described in `RULES.md`.

## Audit

`python -m src.audit strategies/<name>.py` prints a report and exits with 0 on PASS, 1 on REJECT.

| Check | PASS when |
| --- | --- |
| Causality | Weights on prefixes (last full bar, 40, 65, 85, 95, 99 % and 8 random cuts with a fresh seed) equal full-panel weights within 1e-9; repeated run identical; not all zero |
| Data quality | DQS >= 80 |
| Net Sharpe, 5 bps | > 0 on the full panel and on the last 30 % (after a 24h embargo) |
| Lag test | One extra hour of delay keeps Sharpe > 0 |
| Null check | Mean Sharpe over 200 daily-block permutations of returns within +-0.75 |
| Wick check | <= 20 % of positive gross PnL from bars right after liquidation wicks |
| Holdout (maintainer only) | Bar-by-bar weights equal one-shot weights; no crash. Sharpe is reported, not gated |

Fills: signal on the close of bar t, filled at that close. Costs: flat grid 0, 3, 5, 10 bps plus a tiered row (BTC/ETH 5, six least liquid 10, others 7). Funding is charged on the position held across settlement. Every Sharpe has a 95 % block-bootstrap interval: about +-1.5 on this panel, +-6 on the holdout, +-9 on the forward period.

PASS means causal on the tested cuts, reproducible and positive after costs. It is not evidence of an edge: 7 % of random combinations of standard features pass, none with an interval above zero.

The report prints the causality seed. `--seed <n>` reproduces a run; `--n-random-cuts <k>` tests more cuts (the official run uses 60).

Block [13] lists code patterns for manual review. It does not affect the verdict and is easy to evade; every strategy is also read by hand.

## Reference strategies

Verdicts on the full training panel at 5 bps.

| File | Purpose | Sharpe | Verdict |
| --- | --- | --- | --- |
| `baseline_flow.py` | Taker flow + breakout + funding. Designed on the last 144 days, where it scored +3.15 | -1.53 | REJECT |
| `template.py` | Starting point: 24h momentum with deadband | -0.54 | REJECT |
| `leaky_example.py` | Deliberate look-ahead, to show the causality check | -9.21 | REJECT (causality) |

## Common failures

| Report line | Usual cause | Fix |
| --- | --- | --- |
| `[2] ... Look-ahead leak detected` | Full-sample `mean()`/`std()`, `shift(-k)`, `rolling(center=True)` | Rolling or expanding windows only |
| `[2] ... Non-deterministic` | Unseeded randomness | Fix the seed or remove it |
| `[2] ... crashed: FileNotFoundError` | Strategy reads a file | Use only `df` |
| Sharpe at 5 bps <= 0 | Turnover too high | `rebalance_deadband`, slower smoothing |
| `[7] ... FAIL` | Signal decays within an hour | Longer horizon, slower features |
| `[8] ... FAIL` | Sharpe comes from a static tilt | Centre weights per timestamp |
| `[10] ... FAIL` | PnL from wick bars | Skip entries after extreme range bars |

## Limits of the model

- No order book, queue position or market impact; flat taker fees.
- Returns use the last price; exchange funding and margin use the mark price.
- The 20 symbols were chosen in 2026 among contracts liquid today (survivorship).
- Limits of the audit itself: [`LIMITATIONS.md`](LIMITATIONS.md).

## License

Code: MIT ([`LICENSE`](LICENSE)). Market data in `data/`: see [`NOTICE.md`](NOTICE.md).
