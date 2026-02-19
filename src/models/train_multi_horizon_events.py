# src/models/train_multi_horizon_events.py
from __future__ import annotations

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
    feature_version: str | None = None,
    target_version: str | None = None,
) -> EventLogitConfig:
    """
    Central place for multi-horizon defaults.

    Notes:
      - Explicitly passes target_version to prevent mixing versions.
      - Falls back to settings defaults if not provided.
    """
    fv = feature_version or settings.DEFAULT_FEATURE_VERSION
    tv = target_version or settings.DEFAULT_TARGET_VERSION

    if horizon_days == 5:
        return EventLogitConfig(
            market=market,
            horizon_days=5,
            target_name="fwd_ret_5d_log",
            target_version=tv,
            feature_version=fv,
            auto_tune_defaults=True,   # ok to keep on
            test_window_days=90,
            calib_window_days=252,
            calib_split="interleaved",
            event_rule="sigma",
            sigma_mult=1.25,
            allow_isotonic=True,
        )

    if horizon_days == 21:
        return EventLogitConfig(
            market=market,
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
            # cap FIT history to reduce regime mismatch on long horizons
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
) -> None:
    """
    Pipeline-friendly entrypoint:
      - writes artifacts to disk
      - returns None
    """
    for h in horizons:
        cfg = _cfg_for_horizon(
            market=market,
            horizon_days=h,
            feature_version=feature_version,
            target_version=target_version,
        )
        _banner(
            f"EVENT MODEL | market={cfg.market} | horizon={cfg.horizon_days}d "
            f"| target={cfg.target_name} | rule={cfg.event_rule} "
            f"| feat={cfg.feature_version} | tgt_ver={cfg.target_version}"
        )
        run_step6_event_logistic(cfg)


def main() -> None:
    run_multi_horizon_event_suite(
        market="SPY",
        feature_version="v2",
        target_version="v2",
        horizons=(5, 21, 63),
    )


if __name__ == "__main__":
    main()

# Run:
#   python -m src.models.train_multi_horizon_events










