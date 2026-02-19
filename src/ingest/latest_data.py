# src/ingest/latest_data.py
from __future__ import annotations

import datetime as dt
import pandas as pd

from src.ingest.write_raw import read_raw_series_asof


def latest_market_date(
    *,
    series_id: str,
    as_of_ts: dt.datetime,
) -> dt.date:
    """
    Return the latest as_of_date available from the leakage-safe raw read.

    Why this exists:
      - as_of_ts.date() is NOT guaranteed to be a market-close date (weekends/holidays)
      - you might run before today's close is ingested
      - ensures your pipeline's "end_date" matches the newest date actually present in raw

    Inputs
    ------
    series_id:
        The raw series key (e.g., "mkt.spy_close")
    as_of_ts:
        Leakage-safe timestamp used by read_raw_series_asof

    Returns
    -------
    dt.date
        The maximum as_of_date in the raw dataframe.
    """
    df = read_raw_series_asof(series_id=series_id, as_of_ts=as_of_ts)
    if df.empty:
        raise ValueError(
            f"No raw data returned for series_id={series_id} as_of_ts={as_of_ts}."
        )

    # df["as_of_date"] may already be date objects; this is safe either way.
    max_date = pd.to_datetime(df["as_of_date"]).dt.date.max()
    if pd.isna(max_date):
        raise ValueError(f"Could not determine latest as_of_date for series_id={series_id}.")
    return max_date
