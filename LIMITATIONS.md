# Known limits of the audit (v1.8)

The audit is built to catch honest look-ahead mistakes. It is not a security boundary. These gaps are
known; the official verdict protocol in `RULES.md` is designed around them.

## A strategy is ordinary Python

- Each call runs in a separate process from an empty directory, but the process can still read files
  and open network connections. A strategy that reads the dataset from disk can see the future.
- Block [13] of the report flags common patterns (`shift(-k)`, file reads, imports of the harness).
  It is a regex scan and is easy to evade, so it does not affect the verdict.
- The process can also inspect its own call stack and memory. Up to audit v1.1 the bar-by-bar
  evaluation sent the whole holdout or forward period to the strategy process at once, so a strategy
  could read later bars from the evaluator's variables while the leak check still reported no leak.
  Since v1.2 the process is sent one bar at a time and receives the next bar only after it returns
  weights; the canary `tests/fixtures/frame_peek.py` must stay rejected.
- Mitigation: the maintainer reads every strategy file before the official run, and the forward run
  is done with the network off. That covers a participant who tries to read data or call out, but it
  is one person reading, not a barrier. A strategy that hides what it does well enough can get past it.
- With Docker, `--container` is a real barrier: the strategy process has no network, a read-only
  filesystem and no data in the image. CI checks it with the probe `tests/fixtures/escape_probe.py` on
  every push to `main`. Runs without it (CI on pull requests, local runs, and the official run if the
  maintainer has no Docker) get no such isolation, and the report says so.
- Catching this inside Python does not work: native code such as pyarrow reads files without going
  through any Python hook. Also, the parent process holds the whole evaluation period, so a container
  must not share its filesystem or process namespace.
- The strategy also chooses the text of its own errors. Up to audit v1.3.1 that text went into the
  report unchanged: line breaks in it could add lines such as a second `FINAL VERDICT`, and in CI
  close the code block of the job summary. Since v1.3.2 the parent process turns any text that comes
  back from the strategy process into one line of printable ASCII of at most 300 characters, and CI
  prints the report with workflow commands off and HTML-escaped. The verdict, the exit code and the
  `verdict` field of `--log` were never affected; read the verdict from those, or from the line that
  starts with `FINAL VERDICT:`.

## Causality is tested on a finite set of cuts

- Weights are compared on the last full bar, five fixed prefixes and eight random ones. A leak that
  is active only on a few hours can miss all random cuts in a single run, so re-running CI can turn a
  REJECT into a PASS.
- Mitigation: the official run uses 60 random cuts. The seed is printed so any run can be reproduced.
- Two canaries in `tests/fixtures/` (`tail_peek.py`, `known_cuts_peek.py`) reproduce the leaks that
  passed audit v1.0 and must stay rejected.

## CI is not the referee

- A pull request runs the workflow file from its own branch, so a PR that edits `.github/` can change
  what CI checks. Every change to `main` needs the maintainer's approval (`.github/CODEOWNERS`,
  enforced through branch protection on `main`), so participants cannot approve each other's PRs.
- A green CI run is therefore not a verdict; the maintainer's run is.

## Statistics

- PASS means causal, reproducible and positive after costs, not an edge: 7 % of random combinations of
  standard causal features pass, none with a bootstrap interval above zero.
- 95 % interval half-widths are about +-1.5 Sharpe points on the training panel, +-6 on the 36-day
  holdout and +-9 on the 14-day forward period. Final results are not ranked.
- The holdout (Sep 12 - Oct 18) is public Binance data and could have been studied before the freeze.
  Only the forward period is sealed.
- The number of variants a participant tried is not recorded, so the Deflated Sharpe uses an assumed
  number of trials.

## Reporting a new gap

See "Reporting a hole in the audit" in `RULES.md`.
