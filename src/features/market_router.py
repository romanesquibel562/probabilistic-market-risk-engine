# src/features/market_router.py
from __future__ import annotations

from collections.abc import Callable
import pandas as pd

from src.features.feature_defs.close_price_features import compute_close_price_features


FeatureFn = Callable[..., pd.DataFrame]

# Default feature function for any "close price" series
DEFAULT_FEATURE_FN: FeatureFn = compute_close_price_features

# Optional per-market overrides (keep empty until you need special behavior)
MARKET_FEATURE_REGISTRY: dict[str, FeatureFn] = {
    # "SPY": compute_spy_features,  # not needed because default covers it
    # "VIX": compute_vix_features,  # example future override
}

SUPPORTED_MARKETS: set[str] | None = None


def compute_features_for_market(
    market: str,
    prices: pd.DataFrame,
    *,
    price_col: str = "close",
    time_col: str = "as_of_time",
) -> pd.DataFrame:
    """
    Routes markets to feature definition functions.

    Returns long/tall:
      market, <time_col>, feature_name, feature_value

    Notes:
    - By default, any market uses the close-price feature set (log returns, RV, trend, drawdown).
    - Add market-specific overrides in MARKET_FEATURE_REGISTRY when needed.
    """
    mkt = market.upper().strip()

    if SUPPORTED_MARKETS is not None and mkt not in SUPPORTED_MARKETS:
        raise ValueError(
            f"Unsupported market '{mkt}'. Supported: {sorted(SUPPORTED_MARKETS)}"
        )

    fn = MARKET_FEATURE_REGISTRY.get(mkt, DEFAULT_FEATURE_FN)

    # DEFAULT_FEATURE_FN is compute_spy_features but it already accepts market/price_col/time_col
    return fn(
        prices=prices,
        market=mkt,
        price_col=price_col,
        time_col=time_col,
    )
