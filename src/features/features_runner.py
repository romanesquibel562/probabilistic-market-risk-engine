# src/features/features_runner.py
from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Iterable

from src.features.feature_builder import (
    build_and_write_features_asof,
    build_and_write_features_many_asof,
)

# Bump this when you change feature definitions / semantics
FEATURE_VERSION = "v2"

# Default market set (can be expanded)
DEFAULT_MARKETS: list[tuple[str, str]] = [
    ("SPY", "mkt.spy_close"),
    # ("QQQ", "mkt.qqq_close"),
    # ("IWM", "mkt.iwm_close"),
]


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _make_run_id(prefix: str = "features") -> str:
    ts = _utc_now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short}"


def _compute_window(
    *,
    mode: str,
    as_of_ts: dt.datetime,
    lookback_days: int,
    backfill_days: int,
) -> tuple[str, str]:
    """
    Returns (start_date, end_date) as ISO strings for DATE filters.
    """
    end_date = as_of_ts.date().isoformat()

    if mode == "daily":
        start_date = (as_of_ts.date() - dt.timedelta(days=lookback_days)).isoformat()
    elif mode == "backfill":
        start_date = (as_of_ts.date() - dt.timedelta(days=backfill_days)).isoformat()
    else:
        raise ValueError("mode must be 'daily' or 'backfill'")

    return start_date, end_date


def run_features_for_market(
    *,
    market: str,
    series_id: str,
    mode: str = "daily",        # "daily" or "backfill"
    lookback_days: int = 3650,  # used for "daily"
    backfill_days: int = 3650,  # used for "backfill"
    as_of_ts: dt.datetime | None = None,
    feature_version: str = FEATURE_VERSION,
    run_id: str | None = None,
) -> None:
    """
    Single-market features runner.

    IMPORTANT:
    - If you pass as_of_ts from a pipeline, this function will use it verbatim.
    - If run_id is passed, it will be used (so you can unify IDs across pipeline stages).
    """
    if as_of_ts is None:
        as_of_ts = _utc_now()

    if run_id is None:
        run_id = _make_run_id("features_daily" if mode == "daily" else "features_backfill")

    start_date, end_date = _compute_window(
        mode=mode,
        as_of_ts=as_of_ts,
        lookback_days=lookback_days,
        backfill_days=backfill_days,
    )

    feats = build_and_write_features_asof(
        market=market,
        series_id=series_id,
        as_of_ts=as_of_ts,
        run_id=run_id,
        feature_version=feature_version,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"as_of_ts (UTC): {as_of_ts.isoformat()}")
    print(f"Run ID: {run_id}")
    print(f"Mode: {mode}")
    print(f"Feature version: {feature_version}")
    print(f"Wrote {len(feats)} feature rows for {market}.")
    if not feats.empty:
        print(f"Range: {feats['as_of_date'].min()} .. {feats['as_of_date'].max()}")
        print(feats.head(10).to_string(index=False))


def run_features_many(
    *,
    markets: Iterable[tuple[str, str]] = DEFAULT_MARKETS,
    mode: str = "daily",
    lookback_days: int = 3650,
    backfill_days: int = 3650,
    as_of_ts: dt.datetime | None = None,
    feature_version: str = FEATURE_VERSION,
    run_id: str | None = None,
) -> None:
    """
    Multi-market batch runner.

    Same rules:
    - as_of_ts is single source of truth for leakage-safe reads
    - run_id can be injected from pipeline
    """
    if as_of_ts is None:
        as_of_ts = _utc_now()

    if run_id is None:
        run_id = _make_run_id("features_daily" if mode == "daily" else "features_backfill")

    start_date, end_date = _compute_window(
        mode=mode,
        as_of_ts=as_of_ts,
        lookback_days=lookback_days,
        backfill_days=backfill_days,
    )

    feats = build_and_write_features_many_asof(
        markets=markets,
        as_of_ts=as_of_ts,
        run_id=run_id,
        feature_version=feature_version,
        start_date=start_date,
        end_date=end_date,
    )

    markets_list = list(markets)
    print(f"as_of_ts (UTC): {as_of_ts.isoformat()}")
    print(f"Run ID: {run_id}")
    print(f"Mode: {mode}")
    print(f"Feature version: {feature_version}")
    print(f"Markets: {', '.join([m for m, _ in markets_list])}")
    print(f"Wrote {len(feats)} total feature rows.")

    if not feats.empty:
        print(
            feats.groupby("market")["as_of_date"]
            .agg(["min", "max", "count"])
            .rename(columns={"count": "rows"})
            .to_string()
        )
        print(feats.head(10).to_string(index=False))


def run_spy_features(
    *,
    mode: str = "daily",
    lookback_days: int = 3650,
    backfill_days: int = 3650,
    as_of_ts: dt.datetime | None = None,
    run_id: str | None = None,
) -> None:
    """
    Backwards-compatible convenience wrapper.
    """
    return run_features_for_market(
        market="SPY",
        series_id="mkt.spy_close",
        mode=mode,
        lookback_days=lookback_days,
        backfill_days=backfill_days,
        as_of_ts=as_of_ts,
        feature_version=FEATURE_VERSION,
        run_id=run_id,
    )


if __name__ == "__main__":
    # Many markets (recommended)
    run_features_many(mode="daily", lookback_days=550)

    # Run:
    #   python -m src.features.features_runner
    




