# src/decision/exposure_backtest.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class ExposureBacktestConfig:
    preds_csv: str
    output_dir: str
    probability_col: str = "p_sigmoid"
    date_col: str = "as_of_date"
    forward_return_col: str = "y_fwd"

    # Exposure ladder
    low_risk_exposure: float = 1.0
    medium_risk_exposure: float = 0.7
    high_risk_exposure: float = 0.4

    # Percentile thresholds on the chosen probability column
    medium_risk_quantile: float = 0.60
    high_risk_quantile: float = 0.80

    # Annualization factor should match the forecast horizon:
    # 5d  -> 252 / 5  = 50.4
    # 21d -> 252 / 21 = 12
    # 63d -> 252 / 63 = 4
    annualization_factor: float = 252.0 / 5.0

    # Critical fix:
    # Forward returns overlap, so they cannot be cumulatively summed row by row.
    # We therefore sample non-overlapping rows based on the inferred horizon.
    use_non_overlapping_sampling: bool = True

    # Optional override if you want to force a specific step size.
    # If None, it is inferred from annualization_factor.
    non_overlapping_step: int | None = None


# ============================================================
# IO helpers
# ============================================================


def load_predictions(cfg: ExposureBacktestConfig) -> pd.DataFrame:
    """Load and validate the scored predictions file."""
    path = Path(cfg.preds_csv)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)

    required = [cfg.date_col, cfg.forward_return_col, cfg.probability_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    if df[cfg.date_col].isna().any():
        bad_count = int(df[cfg.date_col].isna().sum())
        raise ValueError(f"Failed to parse {bad_count} date values in column '{cfg.date_col}'")

    df[cfg.forward_return_col] = pd.to_numeric(df[cfg.forward_return_col], errors="coerce")
    df[cfg.probability_col] = pd.to_numeric(df[cfg.probability_col], errors="coerce")

    df = df.dropna(subset=[cfg.date_col, cfg.forward_return_col, cfg.probability_col]).copy()
    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain after cleaning the prediction file.")

    return df


# ============================================================
# Exposure rule
# ============================================================


def assign_exposure(df: pd.DataFrame, cfg: ExposureBacktestConfig) -> pd.DataFrame:
    """Map calibrated downside probability to an exposure level."""
    df = df.copy()

    p = df[cfg.probability_col]
    q_med = float(p.quantile(cfg.medium_risk_quantile))
    q_high = float(p.quantile(cfg.high_risk_quantile))

    df["threshold_medium"] = q_med
    df["threshold_high"] = q_high

    conditions = [
        p >= q_high,
        (p >= q_med) & (p < q_high),
        p < q_med,
    ]
    choices = [
        cfg.high_risk_exposure,
        cfg.medium_risk_exposure,
        cfg.low_risk_exposure,
    ]

    df["exposure"] = np.select(conditions, choices, default=cfg.low_risk_exposure).astype(float)

    df["risk_bucket"] = np.where(
        df["exposure"] == cfg.high_risk_exposure,
        "high_risk",
        np.where(df["exposure"] == cfg.medium_risk_exposure, "medium_risk", "low_risk"),
    )

    return df


# ============================================================
# Backtest helpers
# ============================================================


def infer_horizon_step(cfg: ExposureBacktestConfig) -> int:
    """Infer non-overlapping sampling step from annualization factor.

    Since annualization_factor ~= 252 / horizon_days,
    horizon_days ~= 252 / annualization_factor.
    """
    if cfg.non_overlapping_step is not None:
        step = int(cfg.non_overlapping_step)
    else:
        if cfg.annualization_factor <= 0:
            raise ValueError("annualization_factor must be > 0")
        step = int(round(252.0 / cfg.annualization_factor))

    step = max(step, 1)
    return step


def make_non_overlapping(df: pd.DataFrame, cfg: ExposureBacktestConfig) -> pd.DataFrame:
    """Sample non-overlapping rows for horizon-based forward returns.

    This is the critical fix. Forward returns like y_fwd for 63d overlap heavily
    from one row to the next, so summing every row produces impossible equity curves.
    """
    df = df.copy()

    if not cfg.use_non_overlapping_sampling:
        df["sampling_step"] = 1
        df["is_non_overlapping_sample"] = True
        return df.reset_index(drop=True)

    step = infer_horizon_step(cfg)
    sampled = df.iloc[::step].copy().reset_index(drop=True)
    sampled["sampling_step"] = step
    sampled["is_non_overlapping_sample"] = True
    return sampled


# ============================================================
# Backtest math
# ============================================================


def compute_backtest(df: pd.DataFrame, cfg: ExposureBacktestConfig) -> pd.DataFrame:
    """Build the strategy and buy-and-hold return/equity series.

    Assumptions:
    - y_fwd is a forward LOG return over the forecast horizon.
    - strategy_return = exposure * y_fwd
    - equity = exp(cumulative non-overlapping log returns)
    """
    df = make_non_overlapping(df, cfg)

    df["buyhold_return"] = df[cfg.forward_return_col].astype(float)
    df["strategy_return"] = (df["exposure"] * df[cfg.forward_return_col]).astype(float)

    df["buyhold_equity"] = np.exp(df["buyhold_return"].cumsum())
    df["strategy_equity"] = np.exp(df["strategy_return"].cumsum())

    df["buyhold_peak"] = df["buyhold_equity"].cummax()
    df["strategy_peak"] = df["strategy_equity"].cummax()

    df["buyhold_drawdown"] = df["buyhold_equity"] / df["buyhold_peak"] - 1.0
    df["strategy_drawdown"] = df["strategy_equity"] / df["strategy_peak"] - 1.0

    return df


# ============================================================
# Metrics
# ============================================================


def _annualized_return_from_log_returns(log_returns: pd.Series, annualization_factor: float) -> float:
    if len(log_returns) == 0:
        return float("nan")
    mean_log = float(log_returns.mean())
    return float(math.exp(mean_log * annualization_factor) - 1.0)


def _annualized_vol_from_log_returns(log_returns: pd.Series, annualization_factor: float) -> float:
    if len(log_returns) < 2:
        return float("nan")
    return float(log_returns.std(ddof=1) * math.sqrt(annualization_factor))


def _sharpe_from_log_returns(log_returns: pd.Series, annualization_factor: float) -> float:
    vol = _annualized_vol_from_log_returns(log_returns, annualization_factor)
    ret = _annualized_return_from_log_returns(log_returns, annualization_factor)
    if not np.isfinite(vol) or vol == 0.0:
        return float("nan")
    return float(ret / vol)


def _max_drawdown(drawdown: pd.Series) -> float:
    if len(drawdown) == 0:
        return float("nan")
    return float(drawdown.min())


def _calmar(annual_return: float, max_drawdown: float) -> float:
    if not np.isfinite(max_drawdown) or max_drawdown == 0.0:
        return float("nan")
    return float(annual_return / abs(max_drawdown))


def compute_metrics(df: pd.DataFrame, cfg: ExposureBacktestConfig) -> pd.DataFrame:
    strategy_ret = df["strategy_return"]
    buyhold_ret = df["buyhold_return"]

    strategy_ann_ret = _annualized_return_from_log_returns(strategy_ret, cfg.annualization_factor)
    buyhold_ann_ret = _annualized_return_from_log_returns(buyhold_ret, cfg.annualization_factor)

    strategy_ann_vol = _annualized_vol_from_log_returns(strategy_ret, cfg.annualization_factor)
    buyhold_ann_vol = _annualized_vol_from_log_returns(buyhold_ret, cfg.annualization_factor)

    strategy_sharpe = _sharpe_from_log_returns(strategy_ret, cfg.annualization_factor)
    buyhold_sharpe = _sharpe_from_log_returns(buyhold_ret, cfg.annualization_factor)

    strategy_max_dd = _max_drawdown(df["strategy_drawdown"])
    buyhold_max_dd = _max_drawdown(df["buyhold_drawdown"])

    strategy_calmar = _calmar(strategy_ann_ret, strategy_max_dd)
    buyhold_calmar = _calmar(buyhold_ann_ret, buyhold_max_dd)

    avg_exposure = float(df["exposure"].mean())
    pct_reduced = float((df["exposure"] < 1.0).mean())
    pct_high_risk = float((df["exposure"] == cfg.high_risk_exposure).mean())
    pct_medium_risk = float((df["exposure"] == cfg.medium_risk_exposure).mean())

    metrics = pd.DataFrame(
        [
            {
                "strategy": "buy_and_hold",
                "rows": len(df),
                "start_date": df[cfg.date_col].min().date().isoformat(),
                "end_date": df[cfg.date_col].max().date().isoformat(),
                "annual_return": buyhold_ann_ret,
                "annual_vol": buyhold_ann_vol,
                "sharpe": buyhold_sharpe,
                "max_drawdown": buyhold_max_dd,
                "calmar": buyhold_calmar,
                "avg_exposure": 1.0,
                "pct_reduced_exposure": 0.0,
                "pct_high_risk_exposure": 0.0,
                "pct_medium_risk_exposure": 0.0,
                "final_equity": float(df["buyhold_equity"].iloc[-1]),
            },
            {
                "strategy": "probability_scaled",
                "rows": len(df),
                "start_date": df[cfg.date_col].min().date().isoformat(),
                "end_date": df[cfg.date_col].max().date().isoformat(),
                "annual_return": strategy_ann_ret,
                "annual_vol": strategy_ann_vol,
                "sharpe": strategy_sharpe,
                "max_drawdown": strategy_max_dd,
                "calmar": strategy_calmar,
                "avg_exposure": avg_exposure,
                "pct_reduced_exposure": pct_reduced,
                "pct_high_risk_exposure": pct_high_risk,
                "pct_medium_risk_exposure": pct_medium_risk,
                "final_equity": float(df["strategy_equity"].iloc[-1]),
            },
        ]
    )

    return metrics


# ============================================================
# Plotting
# ============================================================


def plot_equity_curve(df: pd.DataFrame, cfg: ExposureBacktestConfig, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df[cfg.date_col], df["buyhold_equity"], label="Buy & Hold")
    plt.plot(df[cfg.date_col], df["strategy_equity"], label="Probability-Scaled Exposure")
    plt.title("Equity Curve (Non-Overlapping Horizon Steps)")
    plt.xlabel("Date")
    plt.ylabel("Equity (starting at 1.0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_drawdown_curve(df: pd.DataFrame, cfg: ExposureBacktestConfig, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df[cfg.date_col], df["buyhold_drawdown"], label="Buy & Hold")
    plt.plot(df[cfg.date_col], df["strategy_drawdown"], label="Probability-Scaled Exposure")
    plt.title("Drawdown Curve (Non-Overlapping Horizon Steps)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# Save outputs
# ============================================================


def save_outputs(df: pd.DataFrame, metrics: pd.DataFrame, cfg: ExposureBacktestConfig) -> Path:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timeseries_path = out_dir / "decision_timeseries.csv"
    metrics_path = out_dir / "decision_metrics.csv"
    config_path = out_dir / "decision_config.json"
    equity_path = out_dir / "equity_curve.png"
    drawdown_path = out_dir / "drawdown_curve.png"

    df.to_csv(timeseries_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    payload = asdict(cfg)
    payload["inferred_non_overlapping_step"] = infer_horizon_step(cfg)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_equity_curve(df, cfg, equity_path)
    plot_drawdown_curve(df, cfg, drawdown_path)

    return out_dir


# ============================================================
# Runner
# ============================================================


def run_exposure_backtest(cfg: ExposureBacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    df = load_predictions(cfg)
    df = assign_exposure(df, cfg)
    df = compute_backtest(df, cfg)
    metrics = compute_metrics(df, cfg)
    out_dir = save_outputs(df, metrics, cfg)
    return df, metrics, out_dir


# ============================================================
# CLI
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Decision layer backtest using scored downside probabilities."
    )
    p.add_argument("--preds-csv", required=True, help="Path to preds_test.csv")
    p.add_argument("--output-dir", required=True, help="Directory to write decision-layer outputs")
    p.add_argument("--probability-col", default="p_sigmoid")
    p.add_argument("--date-col", default="as_of_date")
    p.add_argument("--forward-return-col", default="y_fwd")
    p.add_argument("--low-risk-exposure", type=float, default=1.0)
    p.add_argument("--medium-risk-exposure", type=float, default=0.7)
    p.add_argument("--high-risk-exposure", type=float, default=0.4)
    p.add_argument("--medium-risk-quantile", type=float, default=0.60)
    p.add_argument("--high-risk-quantile", type=float, default=0.80)
    p.add_argument("--annualization-factor", type=float, default=252.0 / 5.0)

    p.add_argument(
        "--disable-non-overlapping-sampling",
        action="store_true",
        help="Disable the non-overlapping horizon sampling fix. Not recommended for forward returns.",
    )
    p.add_argument(
        "--non-overlapping-step",
        type=int,
        default=None,
        help="Optional explicit step size for non-overlapping sampling. "
        "Example: 63 for 63d forward returns.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = ExposureBacktestConfig(
        preds_csv=args.preds_csv,
        output_dir=args.output_dir,
        probability_col=args.probability_col,
        date_col=args.date_col,
        forward_return_col=args.forward_return_col,
        low_risk_exposure=args.low_risk_exposure,
        medium_risk_exposure=args.medium_risk_exposure,
        high_risk_exposure=args.high_risk_exposure,
        medium_risk_quantile=args.medium_risk_quantile,
        high_risk_quantile=args.high_risk_quantile,
        annualization_factor=args.annualization_factor,
        use_non_overlapping_sampling=not args.disable_non_overlapping_sampling,
        non_overlapping_step=args.non_overlapping_step,
    )

    df, metrics, out_dir = run_exposure_backtest(cfg)

    print("\n=== Decision Layer Backtest Complete ===")
    print(f"Outputs written to: {out_dir}")
    print(f"Rows used after sampling: {len(df)}")
    print(f"Non-overlapping step: {infer_horizon_step(cfg)}")
    print("\nMetrics:\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

# python src/decision/exposure_backtest.py --preds-csv "C:\Users\roman\market_risk_engine\artifacts\models\eventlogit_sigma_topk_SPY_fwd_ret_63d_log_h63d_20260212_185622\preds_test.csv" --output-dir "C:\Users\roman\market_risk_engine\artifacts\decision_layer\SPY_63d_v2_fixed" --annualization-factor 4