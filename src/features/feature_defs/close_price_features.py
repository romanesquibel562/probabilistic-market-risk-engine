# src/features/feature_defs/close_price_features.py
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VERSION = "v2"


def compute_market_features(  # <-- new
    prices: pd.DataFrame,
    *,
    market: str,
    price_col: str = "close",
    time_col: str = "as_of_time",
) -> pd.DataFrame:
    """
    Compute leakage-safe daily features for a market from as-of-safe prices.

    Parameters
    ----------
    prices : pd.DataFrame
        Must contain columns [time_col, price_col].

    market : str
        Market label to write into the long output (e.g., "SPY", "QQQ", "BTC").

    Returns
    -------
    pd.DataFrame (long/tall):
        market, <time_col>, feature_name, feature_value
    """
    df = prices[[time_col, price_col]].copy()

    # Normalize time column for stability (works for DATE or datetime inputs)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # Ensure numeric and valid for log
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    df = df[df[price_col] > 0].copy()

    # Log price
    df["logp"] = np.log(df[price_col])

    # Log returns (multi-horizon)
    df["ret_1d_log"] = df["logp"].diff(1)
    df["ret_5d_log"] = df["logp"].diff(5)
    df["ret_21d_log"] = df["logp"].diff(21)
    df["ret_63d_log"] = df["logp"].diff(63)

    # Realized volatility (RMS of 1d log returns)
    df["rv_5d"] = np.sqrt(df["ret_1d_log"].pow(2).rolling(5).mean())
    df["rv_21d"] = np.sqrt(df["ret_1d_log"].pow(2).rolling(21).mean())
    df["rv_63d"] = np.sqrt(df["ret_1d_log"].pow(2).rolling(63).mean())

    eps = 1e-12
    df["rv_ratio_5d_21d"] = df["rv_5d"] / (df["rv_21d"] + eps)
    df["rv_ratio_21d_63d"] = df["rv_21d"] / (df["rv_63d"] + eps)
    df["rv_ratio_5d_63d"] = df["rv_5d"] / (df["rv_63d"] + eps)

    # Trend features (short + medium)
    df["ma_20"] = df[price_col].rolling(20).mean()
    df["px_to_ma20"] = df[price_col] / df["ma_20"] - 1.0

    df["ma_63"] = df[price_col].rolling(63).mean()
    df["px_to_ma63"] = df[price_col] / df["ma_63"] - 1.0

    # Drawdown proxies (21d + 63d rolling max)
    roll_max_21 = df[price_col].rolling(21).max()
    df["dd_21"] = df[price_col] / roll_max_21 - 1.0  # <= 0

    roll_max_63 = df[price_col].rolling(63).max()
    df["dd_63"] = df[price_col] / roll_max_63 - 1.0  # <= 0

    feature_cols = [
        "ret_1d_log",
        "ret_5d_log",
        "ret_21d_log",
        "ret_63d_log",
        "rv_5d",
        "rv_21d",
        "rv_63d",
        "rv_ratio_5d_21d",
        "rv_ratio_21d_63d",
        "rv_ratio_5d_63d",
        "ma_20",
        "px_to_ma20",
        "ma_63",
        "px_to_ma63",
        "dd_21",
        "dd_63",
    ]

    long = df.melt(
        id_vars=[time_col],
        value_vars=feature_cols,
        var_name="feature_name",
        value_name="feature_value",
    )
    long.insert(0, "market", market)

    return long


def compute_close_price_features(
    prices: pd.DataFrame,
    *,
    market: str,
    price_col: str = "close",
    time_col: str = "as_of_time",
) -> pd.DataFrame:
    return compute_market_features(
        prices,
        market=market,
        price_col=price_col,
        time_col=time_col,
    )

