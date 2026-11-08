from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest import coerce_weights

REPO_ROOT = Path(__file__).resolve().parent.parent
CALL_TIMEOUT_SEC = 300
STREAM_TIMEOUT_SEC = 3600
# When set (audit --container), each strategy process runs in this Docker image instead of on the host.
CONTAINER_ENV = "OQS_CONTAINER_IMAGE"


ERROR_TEXT_LIMIT = 300


class StrategyError(RuntimeError):
    pass


def safe_text(text: object, limit: int = ERROR_TEXT_LIMIT) -> str:
    """
    Text that came from a strategy (an error message, the end of its stderr) as one line of printable
    ASCII, at most `limit` characters long. The strategy chooses this text, so it must not add lines to
    the report, carry terminal escape sequences, break the console's encoding or grow without bound.
    Line breaks become " | "; any other character outside printable ASCII is written as its escape
    (\\x1b, \\u202e, ...). Applying it twice gives the same result.
    """
    try:
        raw = str(text)
    except Exception:
        raw = f"<{type(text).__name__} with an unprintable message>"
    joined = " | ".join(part.strip() for part in raw.splitlines() if part.strip())
    out = "".join(c if " " <= c <= "~" else c.encode("unicode_escape").decode("ascii") for c in joined)
    if len(out) > limit:
        marker = " ...[truncated]"
        out = out[: limit - len(marker)] + marker
    return out


def load_module_fn(path: str | Path) -> Callable[[pd.DataFrame], pd.Series]:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "compute_weights"):
        raise AttributeError(f"{path.name} must define `compute_weights(df: pd.DataFrame) -> pd.Series`")
    return getattr(mod, "compute_weights")


def _as_weight_array(df: pd.DataFrame, raw) -> np.ndarray:
    return coerce_weights(df, raw).to_numpy(dtype=np.float64)


def stream_weights(fn: Callable[[pd.DataFrame], pd.Series], df: pd.DataFrame, timestamps) -> np.ndarray:
    """
    Weights for the rows at each timestamp in `timestamps`, computed from df[timestamp <= t] only.
    Calls are made in increasing t, so the strategy never sees a bar later than the one it is weighting.
    Output follows the row order of df restricted to those timestamps.
    """
    ts_col = df["timestamp"]
    out = []
    for t in sorted(pd.to_datetime(list(timestamps), utc=True)):
        sub = df.loc[ts_col <= t].reset_index(drop=True)
        w = _as_weight_array(sub, fn(sub.copy()))
        out.append(w[(sub["timestamp"] == t).to_numpy()])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def _worker_cmd(strategy_path: Path, timeout: int, *extra: str) -> list[str]:
    image = os.environ.get(CONTAINER_ENV)
    if not image:
        return [sys.executable, "-m", "src.sandbox", str(strategy_path), *extra]
    # No network, read-only filesystem, no capabilities; the only host file it can see is the strategy.
    # The image holds src/ and the pinned dependencies, never the data. The inner timeout stops the
    # container itself, which killing the docker client would not.
    return [
        "docker", "run", "-i", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "64",
        "-v", f"{strategy_path}:/work/strategy.py:ro", image,
        "timeout", str(timeout), "python", "-m", "src.sandbox", "/work/strategy.py", *extra,
    ]


def _run_worker(strategy_path: Path, request: dict, timeout: int) -> tuple[np.ndarray, float]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with tempfile.TemporaryDirectory(prefix="oqs_run_") as workdir:
        try:
            proc = subprocess.run(
                _worker_cmd(strategy_path, timeout),
                input=pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL),
                capture_output=True,
                cwd=workdir,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise StrategyError(f"Strategy timed out after {timeout}s") from exc

    payload = proc.stdout
    if payload[:2] == b"OK" and len(payload) > 10:
        (elapsed,) = struct.unpack("<d", payload[2:10])
        arr = np.load(io.BytesIO(payload[10:]), allow_pickle=False)
        if arr.dtype != np.float64 or arr.ndim != 1:
            raise StrategyError("Worker returned malformed weights")
        return arr, float(elapsed)
    # The worker runs the strategy's code, so whatever it sends back is strategy-controlled text.
    if payload[:2] == b"ER":
        raise StrategyError(safe_text(payload[2:].decode("utf-8", errors="replace")))
    if proc.returncode == 124:  # the container's own timeout
        raise StrategyError(f"Strategy timed out after {timeout}s")
    tail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-5:]
    raise StrategyError(f"Strategy process exited with code {proc.returncode}: {safe_text(' | '.join(tail))}")


def isolated_call_timed(strategy_path: str | Path, df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Run compute_weights once in a fresh interpreter whose working directory is an empty temp dir.
    Returns the weights and the time spent inside compute_weights, in seconds.
    """
    return _run_worker(Path(strategy_path).resolve(), {"op": "call", "df": df}, CALL_TIMEOUT_SEC)


def isolated_call(strategy_path: str | Path, df: pd.DataFrame) -> np.ndarray:
    return isolated_call_timed(strategy_path, df)[0]


def _send_frame(stream, payload: bytes) -> None:
    stream.write(struct.pack("<Q", len(payload)) + payload)
    stream.flush()


def _recv_frame(stream) -> bytes | None:
    head = stream.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    data = stream.read(n)
    return data if len(data) == n else None


def isolated_stream(strategy_path: str | Path, df: pd.DataFrame, timestamps) -> np.ndarray:
    """
    Same output as stream_weights, computed in one fresh interpreter that is sent the panel one bar
    at a time. The next bar is written only after the weights for the current one come back, so the
    process never holds a later bar, not even in memory the strategy could reach through the stack.
    """
    ts_col = df["timestamp"]
    if not ts_col.is_monotonic_increasing:
        raise ValueError("isolated_stream needs a panel sorted by timestamp")
    order = sorted(pd.to_datetime(list(timestamps), utc=True))
    if not order:
        return np.zeros(0, dtype=np.float64)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    out = []
    with tempfile.TemporaryDirectory(prefix="oqs_run_") as workdir, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            _worker_cmd(Path(strategy_path).resolve(), STREAM_TIMEOUT_SEC, "--stream"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            cwd=workdir,
            env=env,
        )
        timer = threading.Timer(STREAM_TIMEOUT_SEC, proc.kill)
        timer.start()
        try:
            prev = None
            for t in order:
                chunk = df.loc[(ts_col <= t) & (ts_col > prev)] if prev is not None else df.loc[ts_col <= t]
                prev = t
                try:
                    _send_frame(proc.stdin, pickle.dumps((t, chunk.reset_index(drop=True)), protocol=pickle.HIGHEST_PROTOCOL))
                except OSError:
                    reply = None
                else:
                    reply = _recv_frame(proc.stdout)
                if reply is not None and reply[:2] == b"OK":
                    arr = np.load(io.BytesIO(reply[2:]), allow_pickle=False)
                    if arr.dtype != np.float64 or arr.ndim != 1:
                        raise StrategyError("Worker returned malformed weights")
                    out.append(arr)
                    continue
                if reply is not None and reply[:2] == b"ER":
                    raise StrategyError(safe_text(reply[2:].decode("utf-8", errors="replace")))
                if not timer.is_alive():
                    raise StrategyError(f"Strategy timed out after {STREAM_TIMEOUT_SEC}s")
                proc.wait(timeout=30)
                if proc.returncode == 124:  # the container's own timeout
                    raise StrategyError(f"Strategy timed out after {STREAM_TIMEOUT_SEC}s")
                err.seek(0)
                tail = err.read().decode("utf-8", errors="replace").strip().splitlines()[-5:]
                raise StrategyError(f"Strategy process exited with code {proc.returncode}: {safe_text(' | '.join(tail))}")
        finally:
            timer.cancel()
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return np.concatenate(out)


def _result_channel():
    # The parent reads results from the original stdout; anything the strategy prints goes to stderr.
    result_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(result_fd, "wb")


def _stream_worker_main(strategy_path: str) -> int:
    result = _result_channel()
    panel = None
    try:
        fn = load_module_fn(strategy_path)
        while (frame := _recv_frame(sys.stdin.buffer)) is not None:
            t, chunk = pickle.loads(frame)
            panel = chunk if panel is None else pd.concat([panel, chunk], ignore_index=True)
            w = _as_weight_array(panel, fn(panel.copy()))
            buf = io.BytesIO()
            np.save(buf, np.ascontiguousarray(w[(panel["timestamp"] == t).to_numpy()], dtype=np.float64), allow_pickle=False)
            _send_frame(result, b"OK" + buf.getvalue())
    except BaseException as exc:
        _send_frame(result, b"ER" + f"{type(exc).__name__}: {exc}".encode("utf-8"))
    finally:
        result.close()
    return 0


def _worker_main(strategy_path: str) -> int:
    result = _result_channel()

    try:
        request = pickle.loads(sys.stdin.buffer.read())
        fn = load_module_fn(strategy_path)
        df = request["df"]
        t0 = time.perf_counter()
        if request["op"] == "call":
            w = _as_weight_array(df, fn(df.copy()))
        else:
            raise ValueError(f"Unknown request: {request['op']}")
        elapsed = time.perf_counter() - t0
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(w, dtype=np.float64), allow_pickle=False)
        result.write(b"OK" + struct.pack("<d", elapsed) + buf.getvalue())
    except BaseException as exc:
        result.write(b"ER" + f"{type(exc).__name__}: {exc}".encode("utf-8"))
    finally:
        result.flush()
        result.close()
    return 0


if __name__ == "__main__":
    if sys.argv[2:] == ["--stream"]:
        raise SystemExit(_stream_worker_main(sys.argv[1]))
    raise SystemExit(_worker_main(sys.argv[1]))
