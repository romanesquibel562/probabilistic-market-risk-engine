# src/reporting/export_spy_prices.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingest.connectors.market_prices import fetch_daily_close_stooq


OUTPUT_PATH = Path("artifacts/outputs/spy_prices.csv")


def export_spy_prices() -> None:
    print("\n=== Export SPY Prices ===")

    df = fetch_daily_close_stooq(
        symbol="SPY",
        series_id="SPY",
    )

    if df is None or df.empty:
        raise ValueError("No SPY data returned from fetch_daily_close_stooq().")

    print(f"[export] rows fetched = {len(df)}")

    expected_cols = {"as_of_date", "value"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )

    out = df[["as_of_date", "value"]].copy()
    out = out.rename(columns={"as_of_date": "date", "value": "close"})

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    out = (
        out.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"[export] cleaned rows = {len(out)}")
    print(f"[export] date range    = {out['date'].min()} -> {out['date'].max()}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    export_spy_prices()

    # python -m src.reporting.export_spy_prices


if __name__ == "__main__":
    export_spy_prices()

    # python -m src.reporting.export_spy_prices
