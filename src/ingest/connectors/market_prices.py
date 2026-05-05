from __future__ import annotations

import datetime as dt
from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd
import yfinance as yf


class MarketDataRateLimitError(RuntimeError):
    """Raised when the upstream market-data provider is rate limited."""


_CACHE_DIR = Path(__file__).resolve().parents[3] / "_local_tmp" / "yfinance_tz_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_PRICE_CACHE_DIR = Path(__file__).resolve().parents[3] / "_local_tmp" / "market_data"
_PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    yf.set_tz_cache_location(str(_CACHE_DIR))
except Exception:
    pass


def _available_time_utc_for_daily_close(as_of_date: dt.date) -> datetime:
    """
    Conservative availability timestamp for a US equity daily close.

    21:05 UTC ~= 4:05pm ET (standard time). Intentionally "late" to avoid leakage.
    """
    return datetime(
        as_of_date.year, as_of_date.month, as_of_date.day, 21, 5, 0, tzinfo=timezone.utc
    )


def _normalize_yahoo_symbol(symbol: str) -> str:
    """
    Normalize a symbol for Yahoo Finance usage.

    Examples:
      "SPY" -> "SPY"
      " spy " -> "SPY"
      "spy.us" -> "SPY"
      "SPY.US" -> "SPY"
    """
    s = symbol.strip().upper()

    # Old Stooq-style symbols like SPY.US should become SPY for yfinance
    if s.endswith(".US"):
        s = s[:-3]

    return s


def _price_cache_path(ticker: str) -> Path:
    return _PRICE_CACHE_DIR / f"{ticker.lower()}_daily.csv"


def _load_cached_prices(
    *,
    ticker: str,
    series_id: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    path = _price_cache_path(ticker)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce").dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["available_time"] = pd.to_datetime(df["available_time"], utc=True, errors="coerce")
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["as_of_date", "value"]).copy()

    if start:
        start_d = dt.date.fromisoformat(start)
        df = df[df["as_of_date"] >= start_d]
    if end:
        end_d = dt.date.fromisoformat(end)
        df = df[df["as_of_date"] <= end_d]

    if df.empty:
        return pd.DataFrame()

    df["series_id"] = series_id
    # Keep the canonical source label so cached refreshes dedupe cleanly
    # against prior live Yahoo rows in the warehouse.
    df["source"] = "yfinance"
    df["ingested_at"] = datetime.now(timezone.utc)
    return df[
        ["series_id", "source", "as_of_date", "value", "available_time", "ingested_at"]
    ].sort_values("as_of_date").reset_index(drop=True)


def _write_price_cache(ticker: str, df: pd.DataFrame) -> None:
    path = _price_cache_path(ticker)
    cache_df = df.copy()
    cache_df.to_csv(path, index=False)


def fetch_daily_close_yfinance(
    *,
    symbol: str,
    series_id: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch daily close from Yahoo Finance via yfinance.

    Returns canonical raw schema:
      series_id, source, as_of_date, value, available_time, ingested_at
    """
    ticker = _normalize_yahoo_symbol(symbol)

    last_exc: Exception | None = None
    df = None
    sleep_seconds = [1.0, 2.0, 4.0]

    for attempt, pause_s in enumerate(sleep_seconds, start=1):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                repair=True,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            if "rate limit" in msg or "too many requests" in msg:
                if attempt < len(sleep_seconds):
                    time.sleep(pause_s)
                    continue
                raise MarketDataRateLimitError(
                    f"yfinance rate limited for symbol={ticker}: {e}"
                ) from e
            raise RuntimeError(
                f"yfinance download failed for symbol={ticker}: {e}"
            ) from e

        # yfinance sometimes prints the rate-limit issue and returns an empty frame.
        if attempt < len(sleep_seconds):
            time.sleep(pause_s)

    if df is None or df.empty:
        cached = _load_cached_prices(
            ticker=ticker,
            series_id=series_id,
            start=start,
            end=end,
        )
        if not cached.empty:
            return cached
        if last_exc is not None:
            msg = str(last_exc).lower()
            if "rate limit" in msg or "too many requests" in msg:
                raise MarketDataRateLimitError(
                    f"yfinance rate limited for symbol={ticker}: {last_exc}"
                ) from last_exc
        raise ValueError(
            f"No data returned from yfinance for symbol={ticker}, start={start}, end={end}."
        )

    # Flatten any MultiIndex columns just in case
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    cols_lower = {str(c).lower(): c for c in df.columns}
    close_col = cols_lower.get("close")

    if close_col is None:
        raise ValueError(
            f"Expected a Close column from yfinance for symbol={ticker}. "
            f"Columns found: {list(df.columns)}"
        )

    out = df[[close_col]].copy().reset_index()

    # yfinance may return either 'Date' or a datetime-like index name
    date_col = out.columns[0]
    out = out.rename(columns={date_col: "as_of_date", close_col: "value"})

    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce").dt.date
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["as_of_date", "value"]).sort_values("as_of_date").reset_index(drop=True)

    if out.empty:
        raise ValueError(
            f"yfinance returned rows, but none survived cleaning for symbol={ticker}."
        )

    out["series_id"] = series_id
    out["source"] = "yfinance"
    out["available_time"] = out["as_of_date"].apply(_available_time_utc_for_daily_close)
    out["ingested_at"] = datetime.now(timezone.utc)
    _write_price_cache(ticker, out)

    return out[
        ["series_id", "source", "as_of_date", "value", "available_time", "ingested_at"]
    ].reset_index(drop=True)


# --- Backward compatibility wrappers ------------------------------------------
def fetch_daily_close_stooq(
    *,
    symbol: str,
    series_id: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper so the rest of the pipeline does not need
    to change immediately. Internally uses yfinance now.
    """
    return fetch_daily_close_yfinance(
        symbol=symbol,
        series_id=series_id,
        start=start,
        end=end,
    )


def fetch_spy_daily_stooq(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Backward compatible wrapper for older code paths.
    """
    return fetch_daily_close_yfinance(
        symbol="SPY",
        series_id="mkt.spy_close",
        start=start,
        end=end,
    )



