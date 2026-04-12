# src/reporting/run_evaluation.py
from __future__ import annotations

import argparse
import datetime as dt

from src.reporting.model_evaluator import (
    build_backtest_eval_summary,
    build_model_eval_summary,
    write_backtest_outputs,
    write_eval_outputs,
)
from src.reporting.product_summary import write_product_outputs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="SPY")
    p.add_argument("--as-of-date", default=dt.date.today().isoformat())
    p.add_argument(
        "--latest-only",
        action="store_true",
        help="Keep only latest run per (market,target,horizon,rule) / (market,horizon) for backtests",
    )
    args = p.parse_args()

    # Phase 7: artifact model runs
    df7 = build_model_eval_summary(market=args.market, latest_only=args.latest_only)
    if df7.empty:
        print(f"[run_evaluation] No eventlogit model artifact runs found for market={args.market}")
    else:
        csv_path, md_path = write_eval_outputs(df7, market=args.market, as_of_date=args.as_of_date)
        print("[run_evaluation] Wrote:", csv_path)
        print("[run_evaluation] Wrote:", md_path)

    # Phase 8: walk-forward backtests
    df8 = build_backtest_eval_summary(market=args.market, latest_only=args.latest_only)
    if df8.empty:
        print(f"[run_evaluation] No walk-forward backtest runs found for market={args.market}")
    else:
        csv_path, md_path = write_backtest_outputs(df8, market=args.market, as_of_date=args.as_of_date)
        print("[run_evaluation] Wrote:", csv_path)
        print("[run_evaluation] Wrote:", md_path)

    product_paths = write_product_outputs(
        market=args.market,
        as_of_date=args.as_of_date,
    )
    latest_paths = write_product_outputs(
        market=args.market,
        as_of_date="latest",
    )
    print("[run_evaluation] Wrote:", product_paths.signals_csv)
    print("[run_evaluation] Wrote:", product_paths.history_csv)
    print("[run_evaluation] Wrote:", product_paths.model_health_csv)
    print("[run_evaluation] Wrote:", product_paths.state_json)
    print("[run_evaluation] Refreshed:", latest_paths.state_json)


if __name__ == "__main__":
    main()

# Example:
#   python -m src.reporting.run_evaluation --market SPY --latest-only --as-of-date 2026-03-01

# Example:
#   python -m src.reporting.run_evaluation --market SPY --latest-only --as-of-date 2026-03-01
