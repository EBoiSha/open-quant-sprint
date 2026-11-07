from __future__ import annotations

import pandas as pd
from src.features import cross_sectional_zscore, parkinson_volatility, rebalance_deadband


def compute_weights(df: pd.DataFrame) -> pd.Series:
    """
    Starter template for sprint submissions.
    Input `df` is sorted by (timestamp, symbol) and follows src.schema.REQUIRED_COLS.

    Rules:
    1. Weight at index i must depend ONLY on data at timestamps <= df.loc[i, 'timestamp'].
    2. Never call global .mean(), .std(), or .shift(-k) across the full time axis.
    3. Return a Series strictly aligned with df.index (dollar-neutral cross-sectional weights).
    """
    # 1. Signal: 24h causal trend continuation scaled by range volatility
    p_vol = parkinson_volatility(df, window=24).replace(0.0, float("nan"))
    ret_24h = df.groupby("symbol")["close"].pct_change(24)
    raw_signal = ret_24h / p_vol

    # 2. Cross-sectional dollar-neutral standardization
    z_weights = cross_sectional_zscore(df, raw_signal.fillna(0.0))

    # 3. Causal EWM smoothing + deadband filter to suppress fee drag
    smoothed = z_weights.groupby(df["symbol"]).transform(
        lambda x: x.ewm(span=18, min_periods=1).mean()
    )
    target = cross_sectional_zscore(df, smoothed)
    return rebalance_deadband(df, target, buffer=0.015)
