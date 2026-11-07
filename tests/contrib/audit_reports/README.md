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

f9c2892 is `main` at the time of writing; its `src/audit.py` is that of 6be331d, where version 1.3.1
was set. 6ed3535 is the first commit: it predates the `audit v...` line, the 95 % intervals and the
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
