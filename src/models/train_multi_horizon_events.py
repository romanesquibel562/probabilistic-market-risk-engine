# src/models/train_multi_horizon_events.py
from __future__ import annotations

import datetime as dt

from src.core.config import settings
from src.models.common.risk_event_logistic import EventLogitConfig, run_step6_event_logistic


def _banner(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _cfg_for_horizon(
    *,
    market: str,
    horizon_days: int,
    as_of_ts: dt.datetime,
    feature_version: str | None = None,
    target_version: str | None = None,
) -> EventLogitConfig:
    """
    Central place for multi-horizon defaults.

    Notes:
      - Explicitly passes target_version to prevent mixing versions.
      - as_of_ts is REQUIRED so every horizon sees the same leakage-safe snapshot.
    """
    fv = feature_version or settings.DEFAULT_FEATURE_VERSION
    tv = target_version or settings.DEFAULT_TARGET_VERSION

    if horizon_days == 5:
        return EventLogitConfig(
            market=market,
            as_of_ts=as_of_ts,
            horizon_days=5,
            target_name="fwd_ret_5d_log",
            target_version=tv,
            feature_version=fv,
            auto_tune_defaults=True,
            test_window_days=90,
            calib_window_days=252,
            calib_split="interleaved",
            event_rule="sigma",
            sigma_mult=1.50,
            allow_isotonic=True,
        )

    if horizon_days == 21:
        return EventLogitConfig(
            market=market,
            as_of_ts=as_of_ts,
            horizon_days=21,
            target_name="fwd_ret_21d_log",
            target_version=tv,
            feature_version=fv,
            auto_tune_defaults=True,
            test_window_days=252,
            calib_window_days=252,
            calib_split="interleaved",
            event_rule="sigma",
            sigma_mult=1.0,
            allow_isotonic=False,
        )

    if horizon_days == 63:
        return EventLogitConfig(
            market=market,
            as_of_ts=as_of_ts,
            horizon_days=63,
            target_name="fwd_ret_63d_log",
            target_version=tv,
            feature_version=fv,
            auto_tune_defaults=True,
            test_window_days=504,
            calib_window_days=504,
            calib_split="interleaved",
            event_rule="sigma",
            sigma_mult=1.25,
            allow_isotonic=False,
            fit_max_rows=750,
            fit_tail_only=True,
            debug_splits=True,
        )

    raise ValueError(f"Unsupported horizon_days={horizon_days}. Expected 5, 21, or 63.")


def _cfg_for_horizon_tail_shock(
    *,
    market: str,
    horizon_days: int,
    as_of_ts: dt.datetime,
    feature_version: str | None = None,
    target_version: str | None = None,
    q: float = 0.10,
) -> EventLogitConfig:
    """
    Tail Shock (Quantile) Event:
      Event = 1 if y_fwd <= FIT quantile(q), else 0
    """
    fv = feature_version or settings.DEFAULT_FEATURE_VERSION
    tv = target_version or settings.DEFAULT_TARGET_VERSION

    if horizon_days == 5:
        return EventLogitConfig(
            market=market,
            as_of_ts=as_of_ts,
            horizon_days=5,
            target_name="fwd_ret_5d_log",
            target_version=tv,
            feature_version=fv,
            auto_tune_defaults=True,
            test_window_days=90,
            calib_window_days=252,
            calib_split="interleaved",
            event_rule="quantile",
            event_quantile=q,
            allow_isotonic=True,
        )

    if horizon_days == 21:
        return EventLogitConfig(
            market=market,
            as_of_ts=as_of_ts,
            horizon_days=21,
            target_name="fwd_ret_21d_log",
            target_version=tv,
            feature_version=fv,
            auto_tune_defaults=True,
            test_window_days=252,
            calib_window_days=252,
            calib_split="interleaved",
            event_rule="quantile",
            event_quantile=q,
            allow_isotonic=False,
        )

    if horizon_days == 63:
        return EventLogitConfig(
            market=market,
            as_of_ts=as_of_ts,
            horizon_days=63,
            target_name="fwd_ret_63d_log",
            target_version=tv,
            feature_version=fv,
            auto_tune_defaults=True,
            test_window_days=504,
            calib_window_days=504,
            calib_split="interleaved",
            event_rule="quantile",
            event_quantile=q,
            allow_isotonic=False,
            fit_max_rows=750,
            fit_tail_only=True,
            debug_splits=True,
        )

    raise ValueError(f"Unsupported horizon_days={horizon_days}. Expected 5, 21, or 63.")


def run_multi_horizon_event_suite(
    *,
    market: str = "SPY",
    feature_version: str | None = None,
    target_version: str | None = None,
    horizons: tuple[int, ...] = (5, 21, 63),
    run_sigma: bool = True,
    run_tail_shock: bool = True,
    tail_q: float = 0.10,
    as_of_ts: dt.datetime | None = None,
) -> None:
    """
    Runs sigma + tail-shock (quantile) suites across horizons with ONE pinned as_of_ts.

    If as_of_ts is None, we create it ONCE and reuse it for every horizon/run.
    """
    if as_of_ts is None:
        as_of_ts = dt.datetime.now(dt.timezone.utc)
    else:
        if as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.replace(tzinfo=dt.timezone.utc)
        else:
            as_of_ts = as_of_ts.astimezone(dt.timezone.utc)

    print(f"\nPinned as_of_ts for entire suite: {as_of_ts.isoformat()}")

    for h in horizons:
        if run_sigma:
            cfg = _cfg_for_horizon(
                market=market,
                horizon_days=h,
                as_of_ts=as_of_ts,
                feature_version=feature_version,
                target_version=target_version,
            )
            _banner(
                f"EVENT MODEL | market={cfg.market} | horizon={cfg.horizon_days}d "
                f"| target={cfg.target_name} | rule={cfg.event_rule} "
                f"| feat={cfg.feature_version} | tgt_ver={cfg.target_version}"
            )
            run_step6_event_logistic(cfg)

        if run_tail_shock:
            cfg_q = _cfg_for_horizon_tail_shock(
                market=market,
                horizon_days=h,
                as_of_ts=as_of_ts,
                feature_version=feature_version,
                target_version=target_version,
                q=tail_q,
            )
            _banner(
                f"EVENT MODEL | market={cfg_q.market} | horizon={cfg_q.horizon_days}d "
                f"| target={cfg_q.target_name} | rule={cfg_q.event_rule}(q={tail_q:.2f}) "
                f"| feat={cfg_q.feature_version} | tgt_ver={cfg_q.target_version}"
            )
            run_step6_event_logistic(cfg_q)


def main() -> None:
    run_multi_horizon_event_suite(
        market="SPY",
        feature_version="v2",
        target_version="v2",
        horizons=(5, 21, 63),
        run_sigma=True,
        run_tail_shock=True,
        tail_q=0.10,
    )


if __name__ == "__main__":
    main()

# Run:
#   python -m src.models.train_multi_horizon_events










