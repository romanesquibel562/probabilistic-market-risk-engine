# main.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.pipeline.daily_pipeline import run_daily_pipeline
from src.reporting.export_spy_prices import export_spy_prices
from src.reporting.run_evaluation import main as run_evaluation_main


def _run_streamlit(app_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Market Risk Engine end-to-end and optionally launch the UI."
    )
    parser.add_argument("--market", default="SPY")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-prices-export", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-ui", action="store_true")
    parser.add_argument("--no-ingest", action="store_true")
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--no-targets", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-event-models", action="store_true")
    parser.add_argument("--no-friendly-summary", action="store_true")
    args = parser.parse_args()

    print("\n=== MARKET RISK ENGINE ORCHESTRATOR START ===")
    print(f"market={args.market}")

    if not args.skip_pipeline:
        print("\n[1/4] Running daily pipeline...")
        run_daily_pipeline(
            market=args.market,
            do_ingest=not args.no_ingest,
            do_features=not args.no_features,
            do_targets=not args.no_targets,
            do_validate=not args.no_validate,
            do_event_models=not args.no_event_models,
            do_friendly_summary=not args.no_friendly_summary,
        )
    else:
        print("\n[1/4] Skipping daily pipeline.")

    if not args.skip_prices_export:
        print("\n[2/4] Exporting price history...")
        export_spy_prices()
    else:
        print("\n[2/4] Skipping price export.")

    if not args.skip_evaluation:
        print("\n[3/4] Building evaluation and dashboard outputs...")
        original_argv = sys.argv[:]
        try:
            sys.argv = [
                "run_evaluation",
                "--market",
                args.market,
                "--latest-only",
                "--as-of-date",
                "latest",
            ]
            run_evaluation_main()
        finally:
            sys.argv = original_argv
    else:
        print("\n[3/4] Skipping evaluation.")

    if not args.skip_ui:
        print("\n[4/4] Launching UI...")
        app_path = Path(__file__).resolve().parent / "ui" / "app.py"
        _run_streamlit(app_path)
    else:
        print("\n[4/4] Skipping UI launch.")

    print("\n=== MARKET RISK ENGINE ORCHESTRATOR COMPLETE ===")


if __name__ == "__main__":
    main()


# python main.py --skip-ui
