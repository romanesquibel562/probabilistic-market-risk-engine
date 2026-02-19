# src/ingest/connectors/market_prices.py
from __future__ import annotations

import datetime as dt
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests


def _available_time_utc_for_daily_close(as_of_date: dt.date) -> datetime:
    """
    Conservative availability timestamp for a US equity daily close.

    21:05 UTC ~= 4:05pm ET (standard time). Intentionally "late" to avoid leakage.
    """
    return datetime(
        as_of_date.year, as_of_date.month, as_of_date.day, 21, 5, 0, tzinfo=timezone.utc
    )


def _normalize_stooq_symbol(symbol: str) -> str:
    """
    Normalize a user symbol into a Stooq symbol.

    Examples:
      "SPY" -> "spy.us"
      "spy" -> "spy.us"
      "spy.us" -> "spy.us"
    """
    s = symbol.strip().lower()
    if "." not in s:
        s = f"{s}.us"
    return s


def fetch_daily_close_stooq(
    *,
    symbol: str,
    series_id: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch daily close from Stooq for a given symbol.

    Returns canonical raw schema:
      series_id, source, as_of_date, value, available_time, ingested_at
    """
    sym = _normalize_stooq_symbol(symbol)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.lower() for c in df.columns]

    # Expected columns: date, open, high, low, close, volume
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).sort_values("date")

    if start:
        start_d = dt.date.fromisoformat(start)
        df = df[df["date"] >= start_d]
    if end:
        end_d = dt.date.fromisoformat(end)
        df = df[df["date"] <= end_d]

    out = df[["date", "close"]].rename(
        columns={"date": "as_of_date", "close": "value"}
    ).copy()

    out["series_id"] = series_id
    out["source"] = "stooq"
    out["available_time"] = out["as_of_date"].apply(_available_time_utc_for_daily_close)
    out["ingested_at"] = datetime.now(timezone.utc)

    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])

    return out.reset_index(drop=True)


# --- Backward compatibility wrapper ------------------------------------------
def fetch_spy_daily_stooq(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Backward compatible wrapper for older code paths.
    """
    return fetch_daily_close_stooq(
        symbol="SPY",                 # normalized -> "spy.us"
        series_id="mkt.spy_close",
        start=start,
        end=end,
    )


