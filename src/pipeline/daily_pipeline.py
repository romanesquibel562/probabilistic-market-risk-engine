# src/pipeline/daily_pipeline.py
from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.core.config import settings
from src.ingest.ingest_runner import run_spy_ingest
from src.ingest.latest_data import latest_market_date

from src.features.feature_builder import build_and_write_features_asof
from src.features.validate_features import validate_features_latest

from src.targets.targets_builder import build_and_write_targets_asof, DEFAULT_HORIZONS_DAYS
from src.targets.validate_targets import validate_targets_latest

from src.models.common.baselines import RidgeConfig, run_step5_ridge_with_step55_benchmarks
from src.models.train_multi_horizon_events import run_multi_horizon_event_suite

from src.reporting.friendly_outputs import (
    FriendlySummaryConfig,
    build_friendly_risk_summary_from_artifacts,
)


# ----------------------------
# Helpers / config
# ----------------------------

def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _make_run_id(prefix: str) -> str:
    ts = _utc_now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short}"


@dataclass(frozen=True)
class PipelineVersions:
    """
    Centralized version pins so you never mix stores by accident.
    Defaults come from env via settings, but you can override per run.
    """
    feature_version: str = settings.DEFAULT_FEATURE_VERSION
    target_version: str = settings.DEFAULT_TARGET_VERSION


def _as_list_int(xs: Sequence[int] | Iterable[int]) -> list[int]:
    return [int(x) for x in xs]


def _normalize_as_of_ts(as_of_ts: dt.datetime | None) -> dt.datetime:
    if as_of_ts is None:
        as_of_ts = _utc_now()
    if as_of_ts.tzinfo is None:
        return as_of_ts.replace(tzinfo=dt.timezone.utc)
    return as_of_ts.astimezone(dt.timezone.utc)


# ----------------------------
# Main pipeline
# ----------------------------

def run_daily_pipeline(
    *,
    market: str = "SPY",
    series_id: str = "mkt.spy_close",
    lookback_days: int = 550,
    as_of_ts: dt.datetime | None = None,
    horizons_days: Sequence[int] = DEFAULT_HORIZONS_DAYS,
    versions: PipelineVersions = PipelineVersions(),
    # stages
    do_ingest: bool = True,
    do_features: bool = True,
    do_targets: bool = True,
    do_validate: bool = True,
    # optional modeling stages
    do_ridge_baseline: bool = False,
    do_event_models: bool = True,
    # outputs (recruiter-friendly)
    do_friendly_summary: bool = True,
) -> None:
    """
    One-command pipeline (Airflow-ready later, works now).

    ETL:
      1) Ingest raw series (idempotent)
      2) Build features (v-pinned)
      3) Build targets (v-pinned, multi-horizon)
      4) Validate latest views

    Modeling:
      5) Ridge baseline (optional)
      6) Multi-horizon event suite (optional)

    Outputs:
      7) Recruiter-friendly CSV (optional): artifacts/outputs/friendly_risk_summary_{market}_{as_of_date}.csv

    Important:
      - We compute end_date using latest_market_date(series_id, as_of_ts) so we always anchor on
        the most recent actually-available market date in your warehouse.
    """
    as_of_ts = _normalize_as_of_ts(as_of_ts)

    horizons = _as_list_int(horizons_days)
    horizons = sorted(set(horizons))

    # Use the latest available market date *after ingest*, so we’ll compute it twice:
    # once for logging, and again after ingest in case new rows were written.
    latest_date_pre = latest_market_date(series_id=series_id, as_of_ts=as_of_ts)

    stages = {
        "ingest": do_ingest,
        "features": do_features,
        "targets": do_targets,
        "validate": do_validate,
        "ridge": do_ridge_baseline,
        "event_models": do_event_models,
        "friendly_summary": do_friendly_summary,
    }

    print("=== DAILY PIPELINE START ===")
    print("as_of_ts:", as_of_ts)
    print("market:", market, "| series_id:", series_id)
    print("versions:", {"feature_version": versions.feature_version, "target_version": versions.target_version})
    print("horizons_days:", horizons)
    print("latest_market_date (pre-ingest):", latest_date_pre)
    print("stages:", stages)

    # 1) RAW INGEST
    if do_ingest:
        print("\n[1/7] Ingest raw...")
        if market != "SPY":
            print("WARNING: run_spy_ingest() is SPY-only. Skipping ingest for market != SPY.")
        else:
            run_spy_ingest()

    # Recompute latest date after ingest
    latest_date = latest_market_date(series_id=series_id, as_of_ts=as_of_ts)
    end_date = latest_date.isoformat()
    start_date = (latest_date - dt.timedelta(days=int(lookback_days))).isoformat()
    print("latest_market_date (post-ingest):", latest_date)
    print("window:", start_date, "->", end_date)

    # 2) FEATURES
    if do_features:
        print("\n[2/7] Build features...")
        feat_run_id = _make_run_id("features")
        feats = build_and_write_features_asof(
            market=market,
            series_id=series_id,
            as_of_ts=as_of_ts,
            run_id=feat_run_id,
            feature_version=versions.feature_version,
            start_date=start_date,
            end_date=end_date,
        )
        print(f"Features written: {len(feats)} rows | run_id={feat_run_id}")

    # 3) TARGETS
    if do_targets:
        print("\n[3/7] Build targets...")
        tgt_run_id = _make_run_id("targets")
        tgts = build_and_write_targets_asof(
            market=market,
            series_id=series_id,
            as_of_ts=as_of_ts,
            run_id=tgt_run_id,
            target_version=versions.target_version,
            horizons_days=horizons,
            start_date=start_date,
            end_date=end_date,
        )
        print(f"Targets written: {len(tgts)} rows | run_id={tgt_run_id}")

    # 4) VALIDATION
    if do_validate:
        print("\n[4/7] Validate feature/target stores...")
        validate_features_latest(markets=[market], feature_version=versions.feature_version)
        validate_targets_latest(market=[market], target_version=versions.target_version)

    # 5) RIDGE BASELINE (optional)
    if do_ridge_baseline:
        print("\n[5/7] Ridge baseline...")
        cfg = RidgeConfig(
            market=market,
            target_name="fwd_ret_5d_log",
            horizon_days=5,
            feature_version=versions.feature_version,
            target_version=versions.target_version,
            test_window_days=90,
        )
        run_step5_ridge_with_step55_benchmarks(cfg)

    # 6) EVENT MODELS (optional)
    if do_event_models:
        print("\n[6/7] Multi-horizon event models...")
        run_multi_horizon_event_suite(
            market=market,  # just use the first market for modeling
            feature_version=versions.feature_version,
            horizons=tuple(horizons),
        )

    # 7) FRIENDLY SUMMARY CSV (optional)
    if do_friendly_summary:
        print("\n[7/7] Friendly summary CSV...")
        cfg = FriendlySummaryConfig(market=market)
        out_path = build_friendly_risk_summary_from_artifacts(
            market=market,
            as_of_date=end_date,
            horizons=horizons,
            cfg=cfg,
        )
        print(f"Saved: {out_path.as_posix()}")

    print("\n=== DAILY PIPELINE COMPLETE ===")


if __name__ == "__main__":
    run_daily_pipeline()

    # Run:
    #   python -m src.pipeline.daily_pipeline




