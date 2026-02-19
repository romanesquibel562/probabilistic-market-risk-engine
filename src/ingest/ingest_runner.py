# src/ingest/ingest_runner.py
from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable

import pandas as pd

from src.ingest.connectors.market_prices import fetch_daily_close_stooq
from src.ingest.write_raw import upsert_raw_series


FetchFn = Callable[..., pd.DataFrame]


def _default_start_end(
    start: str | None,
    end: str | None,
    *,
    default_lookback_days: int = 3650,
) -> tuple[str, str]:
    if start is None:
        start = (dt.date.today() - dt.timedelta(days=default_lookback_days)).isoformat()
    if end is None:
        end = dt.date.today().isoformat()
    return start, end


# --- Market ingest registry ---------------------------------------------------
# Store connector params per market. This keeps ingest_runner "multi-market"
# without needing 1 fetch function per ticker.
#
# NOTE: Stooq symbols vary by instrument.
# - US ETFs commonly: "spy.us", "qqq.us", "iwm.us"
MARKET_INGEST_REGISTRY: dict[str, dict[str, str]] = {
    "SPY": {"series_id": "mkt.spy_close", "symbol": "spy.us"},
    # Add more as you implement/verify symbols:
    # "QQQ": {"series_id": "mkt.qqq_close", "symbol": "qqq.us"},
    # "IWM": {"series_id": "mkt.iwm_close", "symbol": "iwm.us"},
}


def run_market_ingest(
    *,
    market: str,
    start: str | None = None,
    end: str | None = None,
    lookback_days: int = 3650,
) -> pd.DataFrame:
    """
    Ingest one market's close series into raw tables.

    Returns the fetched raw rows that were upserted.
    """
    mkt = market.upper().strip()
    if mkt not in MARKET_INGEST_REGISTRY:
        raise ValueError(
            f"Unsupported market '{mkt}'. Supported: {sorted(MARKET_INGEST_REGISTRY)}"
        )

    start, end = _default_start_end(start, end, default_lookback_days=lookback_days)

    meta = MARKET_INGEST_REGISTRY[mkt]
    series_id = meta["series_id"]
    symbol = meta["symbol"]

    # Generic Stooq close fetch (canonical raw schema)
    df = fetch_daily_close_stooq(
        symbol=symbol,
        series_id=series_id,
        start=start,
        end=end,
    )

    upsert_raw_series(df)

    min_d = df["as_of_date"].min() if "as_of_date" in df.columns and len(df) else None
    max_d = df["as_of_date"].max() if "as_of_date" in df.columns and len(df) else None
    print(
        f"[{mkt}] Upserted {len(df)} rows into raw_series_values_v3 (deduped). Range={min_d}..{max_d}"
    )

    return df


def run_markets_ingest(
    markets: Iterable[str],
    *,
    start: str | None = None,
    end: str | None = None,
    lookback_days: int = 3650,
) -> None:
    """Convenience loop to ingest multiple markets."""
    for m in markets:
        run_market_ingest(market=m, start=start, end=end, lookback_days=lookback_days)


# --- Backward compatibility ---------------------------------------------------
def run_spy_ingest(start: str | None = None, end: str | None = None) -> None:
    run_market_ingest(market="SPY", start=start, end=end)


if __name__ == "__main__":
    # Old behavior (still works)
    run_spy_ingest()

    # Examples:
    # python -m src.ingest.ingest_runner
    # python -c "from src.ingest.ingest_runner import run_market_ingest; run_market_ingest(market='SPY')"
    # python -c "from src.ingest.ingest_runner import run_markets_ingest; run_markets_ingest(['SPY','QQQ'])"

