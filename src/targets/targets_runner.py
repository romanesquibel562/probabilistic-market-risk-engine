# src/targets/targets_runner.py
from __future__ import annotations

import datetime as dt
import uuid
from typing import Sequence

from src.core.config import settings
from src.targets.targets_builder import build_and_write_targets_asof, DEFAULT_HORIZONS_DAYS


def _make_run_id(prefix: str = "targets") -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short}"


def run_targets(
    *,
    market: str,
    series_id: str,
    lookback_days: int = 3650,
    horizons_days: Sequence[int] = DEFAULT_HORIZONS_DAYS,
    as_of_ts: dt.datetime | None = None,
    target_version: str | None = None,
) -> None:
    """
    Build + write targets for a given market/series_id.

    target_version:
      - if None, uses settings.DEFAULT_TARGET_VERSION
      - set explicitly only when intentionally writing a new version tag
    """
    if as_of_ts is None:
        as_of_ts = dt.datetime.now(dt.timezone.utc)

    end_date = as_of_ts.date().isoformat()
    start_date = (as_of_ts.date() - dt.timedelta(days=lookback_days)).isoformat()

    run_id = _make_run_id()

    if target_version is None:
        target_version = settings.DEFAULT_TARGET_VERSION

    targets = build_and_write_targets_asof(
        market=market,
        series_id=series_id,
        as_of_ts=as_of_ts,
        run_id=run_id,
        target_version=target_version,
        horizons_days=horizons_days,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"Run ID: {run_id}")
    print(f"market={market} series_id={series_id} target_version={target_version}")
    print(f"Wrote {len(targets)} target rows horizons={list(horizons_days)}.")
    print(f"Range: {targets['as_of_date'].min()} .. {targets['as_of_date'].max()}")
    print(targets.head(10).to_string(index=False))


def run_spy_targets(
    series_id: str = "mkt.spy_close",
    lookback_days: int = 3650,
    horizons_days: Sequence[int] = DEFAULT_HORIZONS_DAYS,
    as_of_ts: dt.datetime | None = None,
    target_version: str | None = None,
) -> None:
    """
    Backwards-compatible convenience wrapper.
    """
    run_targets(
        market="SPY",
        series_id=series_id,
        lookback_days=lookback_days,
        horizons_days=horizons_days,
        as_of_ts=as_of_ts,
        target_version=target_version,
    )


if __name__ == "__main__":
    run_spy_targets()

    # Run:
    #   python -m src.targets.targets_runner

