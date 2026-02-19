from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketSpec:
    market: str          # e.g. "SPY"
    series_id: str       # e.g. "mkt.spy_close"
    asset_class: str = "equities"   


MARKETS: dict[str, MarketSpec] = {
    "SPY": MarketSpec(market="SPY", series_id="mkt.spy_close", asset_class="equities"),
    # Add more as you ingest them:
    # "QQQ": MarketSpec(market="QQQ", series_id="mkt.qqq_close", asset_class="equities"),
    # "IWM": MarketSpec(market="IWM", series_id="mkt.iwm_close", asset_class="equities"),
}


def get_market_spec(market: str) -> MarketSpec:
    key = market.upper().strip()
    if key not in MARKETS:
        raise KeyError(
            f"Unknown market '{market}'. Add it to src/markets/registry.py (MARKETS)."
        )
    return MARKETS[key]