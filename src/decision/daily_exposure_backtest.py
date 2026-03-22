from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Column helpers
# ============================================================

PRED_DATE_CANDIDATES = [
    "as_of_date",
    "date",
    "prediction_date",
    "market_date",
    "ds",
]

PROB_CANDIDATES = [
    "p_sigmoid",
    "p_isotonic",
    "p_raw",
    "p_calib_prod",
    "p_test_prod",
    "p_calib",
    "probability",
    "event_probability",
    "pred_prob",
    "risk_probability",
    "prob",
]

PRICE_DATE_CANDIDATES = [
    "date",
    "trade_date",
    "market_date",
    "ds",
]

CLOSE_CANDIDATES = [
    "close",
    "adj_close",
    "px_close",
    "price",
]


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a {kind} column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# Exposure rule
# ============================================================

def exposure_from_probability(
    p: float,
    high_risk_threshold: float,
    medium_risk_threshold: float,
    low_risk_exposure: float,
    medium_risk_exposure: float,
    high_risk_exposure: float,
) -> float:
    """
    Convert risk probability into portfolio exposure.

    Example:
      p >= 0.60 -> 0.00 exposure
      p >= 0.40 -> 0.50 exposure
      else      -> 1.00 exposure
    """
    if pd.isna(p):
        return low_risk_exposure

    if p >= high_risk_threshold:
        return high_risk_exposure
    if p >= medium_risk_threshold:
        return medium_risk_exposure
    return low_risk_exposure


# ============================================================
# Metrics helpers
# ============================================================

def compute_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def annualized_return(daily_returns: pd.Series, annualization_factor: int = 252) -> float:
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan
    total_return = (1.0 + daily_returns).prod()
    n = len(daily_returns)
    if n == 0 or total_return <= 0:
        return np.nan
    return total_return ** (annualization_factor / n) - 1.0


def annualized_vol(daily_returns: pd.Series, annualization_factor: int = 252) -> float:
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return np.nan
    return daily_returns.std(ddof=1) * np.sqrt(annualization_factor)


def sharpe_ratio(
    daily_returns: pd.Series,
    annualization_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return np.nan

    rf_daily = risk_free_rate / annualization_factor
    excess = daily_returns - rf_daily
    vol = excess.std(ddof=1)
    if vol == 0 or pd.isna(vol):
        return np.nan
    return excess.mean() / vol * np.sqrt(annualization_factor)


def sortino_ratio(
    daily_returns: pd.Series,
    annualization_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return np.nan

    rf_daily = risk_free_rate / annualization_factor
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) < 2:
        return np.nan

    downside_std = downside.std(ddof=1)
    if downside_std == 0 or pd.isna(downside_std):
        return np.nan

    return excess.mean() / downside_std * np.sqrt(annualization_factor)


def max_drawdown(daily_returns: pd.Series) -> float:
    equity = (1.0 + daily_returns.fillna(0.0)).cumprod()
    dd = compute_drawdown(equity)
    return dd.min()


def calmar_ratio(daily_returns: pd.Series, annualization_factor: int = 252) -> float:
    ann_ret = annualized_return(daily_returns, annualization_factor)
    mdd = max_drawdown(daily_returns)
    if pd.isna(mdd) or mdd == 0:
        return np.nan
    return ann_ret / abs(mdd)


def summarize_strategy(
    name: str,
    df: pd.DataFrame,
    returns_col: str,
    exposure_col: Optional[str],
    annualization_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    r = df[returns_col].dropna()

    if len(r) == 0:
        return {
            "strategy": name,
            "rows": 0,
            "start_date": pd.NaT,
            "end_date": pd.NaT,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "final_equity": np.nan,
            "avg_exposure": np.nan if exposure_col else 1.0,
            "pct_zero_exposure": np.nan if exposure_col else 0.0,
            "pct_reduced_exposure": np.nan if exposure_col else 0.0,
            "pct_full_exposure": np.nan if exposure_col else 1.0,
        }

    out = {
        "strategy": name,
        "rows": int(len(r)),
        "start_date": df.loc[r.index, "date"].min(),
        "end_date": df.loc[r.index, "date"].max(),
        "annual_return": annualized_return(r, annualization_factor),
        "annual_vol": annualized_vol(r, annualization_factor),
        "sharpe": sharpe_ratio(r, annualization_factor, risk_free_rate),
        "sortino": sortino_ratio(r, annualization_factor, risk_free_rate),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r, annualization_factor),
        "final_equity": (1.0 + r).cumprod().iloc[-1],
    }

    if exposure_col is not None:
        exp = df[exposure_col]
        out["avg_exposure"] = exp.mean()
        out["pct_zero_exposure"] = (exp == 0.0).mean()
        out["pct_reduced_exposure"] = ((exp > 0.0) & (exp < 1.0)).mean()
        out["pct_full_exposure"] = (exp == 1.0).mean()
    else:
        out["avg_exposure"] = 1.0
        out["pct_zero_exposure"] = 0.0
        out["pct_reduced_exposure"] = 0.0
        out["pct_full_exposure"] = 1.0

    return out


# ============================================================
# Core backtest
# ============================================================

def run_daily_backtest(
    preds_csv: str,
    prices_csv: str,
    output_dir: str,
    high_risk_threshold: float = 0.60,
    medium_risk_threshold: float = 0.40,
    low_risk_exposure: float = 1.00,
    medium_risk_exposure: float = 0.50,
    high_risk_exposure: float = 0.00,
    annualization_factor: int = 252,
    transaction_cost_bps: float = 0.0,
    risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(preds_csv)
    prices = pd.read_csv(prices_csv)

    pred_date_col = pick_first_existing_column(preds, PRED_DATE_CANDIDATES, "prediction date")
    prob_col = pick_first_existing_column(preds, PROB_CANDIDATES, "probability")
    price_date_col = pick_first_existing_column(prices, PRICE_DATE_CANDIDATES, "price date")
    close_col = pick_first_existing_column(prices, CLOSE_CANDIDATES, "close")

    preds = preds[[pred_date_col, prob_col]].copy()
    preds.columns = ["date", "risk_prob"]

    prices = prices[[price_date_col, close_col]].copy()
    prices.columns = ["date", "close"]

    preds["date"] = pd.to_datetime(preds["date"], errors="coerce")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    preds["risk_prob"] = pd.to_numeric(preds["risk_prob"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")

    preds = (
        preds.dropna(subset=["date", "risk_prob"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    prices = (
        prices.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    if preds.empty:
        raise ValueError("Predictions dataframe is empty after cleaning.")
    if prices.empty:
        raise ValueError("Prices dataframe is empty after cleaning.")

    # Daily simple returns
    prices["daily_ret"] = prices["close"].pct_change()

    # Merge on daily dates
    df = prices.merge(preds, on="date", how="left").sort_values("date").reset_index(drop=True)

    # Restrict to true prediction window
    first_pred_date = preds["date"].min()
    last_pred_date = preds["date"].max()

    df = df[(df["date"] >= first_pred_date) & (df["date"] <= last_pred_date)].copy()

    # Forward-fill only inside the valid prediction window
    df["risk_prob"] = df["risk_prob"].ffill()

    # Drop any rows that still do not have a usable probability
    df = df.dropna(subset=["risk_prob"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Backtest dataframe is empty after restricting to prediction window. "
            "Check date alignment between predictions and prices."
        )

    # Exposure rule based on same-day signal
    df["target_exposure"] = df["risk_prob"].apply(
        lambda p: exposure_from_probability(
            p=p,
            high_risk_threshold=high_risk_threshold,
            medium_risk_threshold=medium_risk_threshold,
            low_risk_exposure=low_risk_exposure,
            medium_risk_exposure=medium_risk_exposure,
            high_risk_exposure=high_risk_exposure,
        )
    )

    # Apply signal on next trading day to avoid lookahead bias
    df["applied_exposure"] = df["target_exposure"].shift(1)
    df["applied_exposure"] = df["applied_exposure"].fillna(low_risk_exposure)

    # Transaction costs based on turnover
    df["turnover"] = df["applied_exposure"].diff().abs().fillna(0.0)
    df["transaction_cost"] = df["turnover"] * (transaction_cost_bps / 10000.0)

    # Strategy returns
    df["buyhold_ret"] = df["daily_ret"]
    df["strategy_ret_gross"] = df["applied_exposure"] * df["daily_ret"]
    df["strategy_ret_net"] = df["strategy_ret_gross"] - df["transaction_cost"]

    # Equity curves
    df["buyhold_equity"] = (1.0 + df["buyhold_ret"].fillna(0.0)).cumprod()
    df["strategy_equity_gross"] = (1.0 + df["strategy_ret_gross"].fillna(0.0)).cumprod()
    df["strategy_equity_net"] = (1.0 + df["strategy_ret_net"].fillna(0.0)).cumprod()

    # Drawdowns
    df["buyhold_drawdown"] = compute_drawdown(df["buyhold_equity"])
    df["strategy_drawdown_net"] = compute_drawdown(df["strategy_equity_net"])

    metrics = pd.DataFrame(
        [
            summarize_strategy(
                name="buy_and_hold",
                df=df,
                returns_col="buyhold_ret",
                exposure_col=None,
                annualization_factor=annualization_factor,
                risk_free_rate=risk_free_rate,
            ),
            summarize_strategy(
                name="strategy_gross",
                df=df,
                returns_col="strategy_ret_gross",
                exposure_col="applied_exposure",
                annualization_factor=annualization_factor,
                risk_free_rate=risk_free_rate,
            ),
            summarize_strategy(
                name="strategy_net",
                df=df,
                returns_col="strategy_ret_net",
                exposure_col="applied_exposure",
                annualization_factor=annualization_factor,
                risk_free_rate=risk_free_rate,
            ),
        ]
    )

    # Save outputs
    daily_out = output_path / "daily_exposure_backtest.csv"
    metrics_out = output_path / "metrics.csv"

    df.to_csv(daily_out, index=False)
    metrics.to_csv(metrics_out, index=False)

    # Plots
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["buyhold_equity"], label="Buy & Hold")
    plt.plot(df["date"], df["strategy_equity_net"], label="Strategy (Net)")
    plt.plot(df["date"], df["strategy_equity_gross"], label="Strategy (Gross)", alpha=0.7)
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "equity_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["buyhold_drawdown"], label="Buy & Hold Drawdown")
    plt.plot(df["date"], df["strategy_drawdown_net"], label="Strategy Drawdown (Net)")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "drawdown.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(df["date"], df["risk_prob"], label="Risk Probability")
    plt.axhline(high_risk_threshold, linestyle="--", label="High Risk Threshold")
    plt.axhline(medium_risk_threshold, linestyle="--", label="Medium Risk Threshold")
    plt.title("Risk Probability Through Time")
    plt.xlabel("Date")
    plt.ylabel("Predicted Risk Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "risk_probability.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.step(df["date"], df["applied_exposure"], where="post", label="Applied Exposure")
    plt.title("Applied Exposure Through Time")
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "applied_exposure.png", dpi=150)
    plt.close()

    print("\n=== Daily Exposure Backtest Complete ===")
    print(f"Outputs written to: {output_path}")

    print("\nDetected columns:")
    print(f"  preds date column:   {pred_date_col}")
    print(f"  preds prob column:   {prob_col}")
    print(f"  prices date column:  {price_date_col}")
    print(f"  prices close column: {close_col}")

    print("\nBacktest window:")
    print(f"  first prediction date: {first_pred_date.date()}")
    print(f"  last prediction date:  {last_pred_date.date()}")
    print(f"  rows used:             {len(df)}")

    print("\nMetrics:\n")
    print(metrics.to_string(index=False))

    return df, metrics


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily exposure backtest using model risk probabilities.")

    parser.add_argument("--preds-csv", required=True, help="Path to prediction CSV.")
    parser.add_argument("--prices-csv", required=True, help="Path to daily prices CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for results.")

    parser.add_argument("--high-risk-threshold", type=float, default=0.60)
    parser.add_argument("--medium-risk-threshold", type=float, default=0.40)

    parser.add_argument("--low-risk-exposure", type=float, default=1.00)
    parser.add_argument("--medium-risk-exposure", type=float, default=0.50)
    parser.add_argument("--high-risk-exposure", type=float, default=0.00)

    parser.add_argument("--annualization-factor", type=int, default=252)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_daily_backtest(
        preds_csv=args.preds_csv,
        prices_csv=args.prices_csv,
        output_dir=args.output_dir,
        high_risk_threshold=args.high_risk_threshold,
        medium_risk_threshold=args.medium_risk_threshold,
        low_risk_exposure=args.low_risk_exposure,
        medium_risk_exposure=args.medium_risk_exposure,
        high_risk_exposure=args.high_risk_exposure,
        annualization_factor=args.annualization_factor,
        transaction_cost_bps=args.transaction_cost_bps,
        risk_free_rate=args.risk_free_rate,
    )

    # python -m src.decision.daily_exposure_backtest --preds-csv "C:\Users\roman\market_risk_engine\artifacts\models\eventlogit_sigma_topk_SPY_fwd_ret_63d_log_h63d_20260212_185622\preds_test.csv" --prices-csv "C:\Users\roman\market_risk_engine\artifacts\outputs\spy_prices.csv" --output-dir "C:\Users\roman\market_risk_engine\artifacts\decision_layer\daily_backtest_SPY_63d" --high-risk-threshold 0.60 --medium-risk-threshold 0.40 --low-risk-exposure 1.0 --medium-risk-exposure 0.5 --high-risk-exposure 0.0 --annualization-factor 252 --transaction-cost-bps 2