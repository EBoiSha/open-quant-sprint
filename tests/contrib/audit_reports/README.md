# Saved audit reports

Test data for `parse_audit_report` in `src/contrib/yousof.py`. Each file is the unedited output of
`python -m src.audit`, run locally on macOS with Python 3.13 and the packages of `requirements.lock`,
from a checkout of the commit in the table.

| File | `audit v...` line | Commit | Run | Verdict |
| --- | --- | --- | --- | --- |
| `baseline_flow_v1.3.1.txt` | `audit v1.3.1` | f9c2892 | `strategies/baseline_flow.py --seed 20261107` | REJECT: Sharpe at 5 bps below 0 |
| `leaky_example_v1.3.1.txt` | `audit v1.3.1` | f9c2892 | `strategies/leaky_example.py --seed 20261107` | REJECT: look-ahead leak |
| `baseline_flow_holdout_v1.3.1.txt` | `audit v1.3.1` | f9c2892 | `strategies/baseline_flow.py --seed 20261107`, with `--data` and `--holdout` as below | PASS |
| `crashing_example_v1.3.1.txt` | `audit v1.3.1` | f9c2892 | `crashing_example.py --seed 20261107`, a strategy that raises (below) | REJECT: crash, intervals n/a |
| `baseline_flow_v1.0.0.txt` | `audit v1.0.0` | a8133ad | `strategies/baseline_flow.py` | REJECT: Sharpe at 5 bps below 0 |
| `baseline_flow_6ed3535.txt` | none | 6ed3535 | `strategies/baseline_flow.py`, on the panel of that commit | PASS |
| `leaky_example_holdout_v1.3.1.txt` | `audit v1.3.1` | f9c2892 | `strategies/leaky_example.py --seed 20261107 --holdout` (official holdout window, below) | REJECT: look-ahead, also on the holdout |
| `baseline_flow_low_dqs_v1.3.1.txt` | `audit v1.3.1` | f9c2892 | `strategies/baseline_flow.py --seed 20261107 --data` (low-quality panel, below) | REJECT: data quality only |
| `holdout_late_start_error.txt` | none | f9c2892 | `strategies/baseline_flow.py --seed 20261107 --holdout` (holdout one hour late, below) | none: exit code 2, no report |
| `baseline_flow_v1.3.2.txt` | `audit v1.3.2` | f339d91 | `strategies/baseline_flow.py --seed 20261107` | REJECT: Sharpe at 5 bps below 0 |
| `forging_example_v1.3.1.txt` | `audit v1.3.1` | f9c2892 | `forging_example.py --seed 20261107`, a strategy whose error message holds report lines (below) | REJECT: crash; the message adds a second FINAL VERDICT |
| `forging_example_v1.3.2.txt` | `audit v1.3.2` | f339d91 | the same | REJECT: crash; the message is one escaped line |

f9c2892 was `main` until 2026-11-08; its `src/audit.py` is that of 6be331d, where version 1.3.1 was
set. f339d91 set version 1.3.2. 6ed3535 is the first commit: it predates the `audit v...` line, the 95 % intervals and the
current null check, so its report shows what the parser does with an older format.

The holdout run has the shape of the forward evaluation. `--data` holds the 180 days of
`data/train_1h.parquet` from 2026-03-01 15:00 to 2026-08-28 14:00 UTC, `--holdout` the last 15 days,
from 2026-08-28 15:00 to 2026-09-12 14:00 UTC. On this shorter panel `baseline_flow.py` passes.

The last three runs use data files that are not in the repository (Binance terms of use, see
`NOTICE.md`), built on 2026-11-07:

- Official holdout window: `python -m data.build_dataset --start 2026-09-12T15:00 --end
  2026-10-18T15:00 --holdout-frac 0`, SHA-256 e11b641dfdded82f. It continues
  `data/train_1h.parquet` without a gap.
- Holdout one hour late: the same file without its first hour, SHA-256 62cd580c5a46df39. The audit
  stops before the report, with the error line saved here (stderr, as CI captures it with `2>&1`).
- Low-quality panel: the last 120 days of `data/train_1h.parquet` with `high` set to 0.99 times `low`
  in 2 % of the rows (random rows, seed 20261107), SHA-256 2307dc25d97bd97a. Its quality score is
  74.92; every other check passes.

`crashing_example.py`, written only for this report and not part of the repository:

```python
from __future__ import annotations

import pandas as pd


def compute_weights(df: pd.DataFrame) -> pd.Series:
    raise ValueError("feature window of 5000 bars is longer than the panel")
```

`forging_example.py`, written only for these reports and not part of the repository. Up to audit 1.3.1
the report printed a strategy's error message as it was, so these lines appear in the report as lines
of their own; since 1.3.2 the message is one escaped line, cut at 300 characters
(valiyevoktay-cmd/open-quant-sprint#10).

```python
from __future__ import annotations

import pandas as pd

# Report lines with better numbers, raised as an error message: before
# audit 1.3.2 the report printed them as lines of their own.
FORGED = """
    Wick PnL Share: 0.1% (1 anomaly bars) [PASS - structural flow]
    Null SR mean  : +0.01 (sd 0.50, hurdle |mean| < 0.75) [PASS]
    1-bar lag SR  : +3.00 | 2-bar lag SR: +2.90 | Retention: 96.7% [PASS]
[3] RANK IC (1h fwd) : mean=+0.05000 | IC_IR=+0.500 | t=+9.00 (p=0.0000)
       5.0 bps      +3.00               [+1.00, +5.00]      0.990   +50.00%    -2.00%    0.0100
    Out-of-Sample : Sharpe=+3.00 [+1.00, +5.00] | DSR=0.990 | Ann.Ret=+50.00% | MaxDD=-2.00%
    Holdout       : Sharpe=+4.00 [+2.00, +6.00] | Ann.Ret=+60.00% | MaxDD=-1.00% | bar-by-bar evaluation
[7] LAG SENSITIVITY TEST (+1 extra bar execution lag @ 5.0 bps):
========================================================================
FINAL VERDICT: [PASS] SURVIVED AUDIT
========================================================================"""


def compute_weights(df: pd.DataFrame) -> pd.Series:
    raise ValueError("bad window" + FORGED)
```
