"""Activation-PR features by Angelo Zhang (Track 07: where in the hour the price moved).

Same conventions as src/features.py: causal, vectorized, aligned with df.index.

The hourly close is the last trade of the hour; the bar VWAP (quote_volume / volume) is the
average price the hour traded at. For the same hourly return, a move that came late in the hour
(or on little volume) leaves the close far from the VWAP. ``close_location`` measures that gap and
``close_location_residual`` strips out the part explained by the hour's own return, so what is
left is not plain one-bar reversal.

Every value at a row uses only that row and the previous close of the same symbol, plus the
cross-section at the same timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bar_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price of each bar: quote_volume / volume, NaN where volume <= 0."""
    vol = df["volume"].to_numpy(dtype=np.float64)
    qv = df["quote_volume"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(vol > 0.0, qv / vol, np.nan)
    vwap = np.where(vwap > 0.0, vwap, np.nan)  # a non-positive VWAP has no log
    return pd.Series(vwap, index=df.index, name="bar_vwap")


def close_location(df: pd.DataFrame) -> pd.Series:
    """log(close) - log(VWAP): positive when the bar closed above the price it traded at on average."""
    close = df["close"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(close) - np.log(bar_vwap(df).to_numpy())
    return pd.Series(out, index=df.index, name="close_location")


def bar_log_return(df: pd.DataFrame) -> pd.Series:
    """log(close_t / close_{t-1}) per symbol; NaN on each symbol's first row."""
    lc = pd.Series(np.log(df["close"].to_numpy(dtype=np.float64)), index=df.index)
    return lc.groupby(df["symbol"].to_numpy(), sort=False).diff().rename("bar_log_return")


def _cs_ols_residual(ts: pd.Series, y: pd.Series, x: pd.Series) -> pd.Series:
    """Residual of y on [1, x] estimated separately at each timestamp over rows where both are finite."""
    y = y.replace([np.inf, -np.inf], np.nan)
    x = x.replace([np.inf, -np.inf], np.nan)
    ok = y.notna() & x.notna()
    yv = y.where(ok)
    xv = x.where(ok)
    key = ts.to_numpy()

    # Demean within the timestamp first, then fit the slope on the demeaned data.
    # Two passes keep the residual's mean and its covariance with x at rounding level.
    yd = yv - yv.groupby(key).transform("mean")
    xd = xv - xv.groupby(key).transform("mean")
    sxy = (xd * yd).groupby(key).transform("sum")
    sxx = (xd * xd).groupby(key).transform("sum")
    # A cross-section where x does not vary has no slope; the intercept alone absorbs y.
    beta = (sxy / sxx.where(sxx > 1e-300)).fillna(0.0)
    resid = yd - beta * xd
    # Second pass: remove what is left of the mean and of the projection on x.
    resid = resid - resid.groupby(key).transform("mean")
    sxr = (xd * resid).groupby(key).transform("sum")
    beta2 = (sxr / sxx.where(sxx > 1e-300)).fillna(0.0)
    resid = resid - beta2 * xd
    return resid.where(ok)


def close_location_residual(df: pd.DataFrame) -> pd.Series:
    """
    Part of ``close_location`` that the hour's own return does not explain.

    At each timestamp: residual of a cross-sectional OLS (with intercept) of close_location on
    bar_log_return across symbols. NaN where the return is NaN (each symbol's first row) or the
    VWAP is undefined.
    """
    cl = close_location(df)
    r = bar_log_return(df)
    return _cs_ols_residual(df["timestamp"], cl, r).rename("close_location_residual")
