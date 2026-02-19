# src/models/common/baselines.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.training_matrix import build_training_matrix


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class RidgeConfig:
    # Make this explicit so we stop thinking "SPY by default"
    market: str = "SPY"  # you can later remove default and force user to pass it
    target_name: str = "fwd_ret_5d_log"
    horizon_days: int = 5
    feature_version: str = "v2"
    target_version: str = "v2"
    alpha: float = 1.0

    # row-count window (trading days-ish)
    test_window_days: int = 90

    artifacts_root: Path = Path("artifacts/models")


# ============================================================
# Metrics
# ============================================================

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _directional_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true > 0) == (y_pred > 0)))


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "corr": _safe_corr(y_true, y_pred),
        "directional_acc": _directional_acc(y_true, y_pred),
    }


def _utc_run_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


# ============================================================
# Core
# ============================================================

def _build_split_last_n(df: pd.DataFrame, test_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "as_of_date" not in df.columns:
        raise ValueError("Expected 'as_of_date' column in training matrix.")

    df = df.sort_values("as_of_date").reset_index(drop=True)

    if len(df) < (test_n + 50):
        raise ValueError(f"Not enough rows for split. rows={len(df)} needs >= {test_n+50}")

    test = df.iloc[-test_n:].copy()
    train = df.iloc[:-test_n].copy()

    if len(train) < 50:
        raise ValueError(f"Train set too small ({len(train)} rows). Reduce test_window_days.")
    if len(test) < 20:
        raise ValueError(f"Test set too small ({len(test)} rows). Reduce test_window_days.")

    return train, test


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    # As you go multi-market, training_matrix may include more metadata columns.
    exclude = {
        "market",
        "as_of_date",
        "target_value",
        "target_name",
        "horizon_days",
        "feature_version",
        "target_version",
        "series_id",
        "source",
        "available_time",
        "computed_at",
        "run_id",
    }
    cols = [c for c in df.columns if c not in exclude]
    if not cols:
        raise ValueError("No feature columns found after excluding metadata columns.")
    return cols


def _auto_tune_ridge_defaults(cfg: RidgeConfig) -> RidgeConfig:
    # clean + safe for frozen dataclass
    if cfg.horizon_days == 5:
        return replace(cfg, test_window_days=90)
    if cfg.horizon_days == 21:
        return replace(cfg, test_window_days=252)
    if cfg.horizon_days == 63:
        return replace(cfg, test_window_days=504)
    return cfg


def run_step5_ridge_with_step55_benchmarks(cfg: RidgeConfig) -> None:
    print("\n=== STEP 5: Ridge Baseline + STEP 5.5 Naive Benchmarks ===")
    cfg = _auto_tune_ridge_defaults(cfg)

    df = build_training_matrix(
        market=cfg.market,
        target_name=cfg.target_name,
        horizon_days=cfg.horizon_days,
        feature_version=cfg.feature_version,
        target_version=cfg.target_version,
        dropna_features=True,
    )
    if df.empty:
        raise ValueError("Training matrix is empty.")

    train, test = _build_split_last_n(df, test_n=int(cfg.test_window_days))
    feature_cols = _get_feature_cols(train)

    X_train = train[feature_cols].to_numpy()
    y_train = train["target_value"].to_numpy()

    X_test = test[feature_cols].to_numpy()
    y_test = test["target_value"].to_numpy()

    ridge_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=cfg.alpha)),
        ]
    )
    ridge_model.fit(X_train, y_train)

    pred_ridge_train = ridge_model.predict(X_train)
    pred_ridge_test = ridge_model.predict(X_test)

    # Naive baselines
    pred_zero_train = np.zeros_like(y_train)
    pred_zero_test = np.zeros_like(y_test)

    mu = float(np.mean(y_train))
    pred_mean_train = np.full_like(y_train, fill_value=mu, dtype=float)
    pred_mean_test = np.full_like(y_test, fill_value=mu, dtype=float)

    metrics_rows = []
    for name, ytr_hat, yte_hat in [
        ("ridge", pred_ridge_train, pred_ridge_test),
        ("naive_zero", pred_zero_train, pred_zero_test),
        ("naive_train_mean", pred_mean_train, pred_mean_test),
    ]:
        metrics_rows.append({"model": name, "split": "train", **_evaluate(y_train, ytr_hat)})
        metrics_rows.append({"model": name, "split": "test", **_evaluate(y_test, yte_hat)})

    metrics_df = pd.DataFrame(metrics_rows)
    test_metrics = metrics_df[metrics_df["split"] == "test"].copy().sort_values("rmse", ascending=True)

    print(f"Market: {cfg.market} | Target: {cfg.target_name} | Horizon: {cfg.horizon_days}d")
    print(f"Rows total: {len(df)} | Train: {len(train)} | Test: {len(test)}")
    print("\n--- Test Metrics (sorted by RMSE, lower is better) ---")
    print(test_metrics[["model", "mae", "rmse", "r2", "corr", "directional_acc"]].to_string(index=False))

    ridge = ridge_model.named_steps["ridge"]
    coef_df = (
        pd.DataFrame({"feature": feature_cols, "coef": ridge.coef_})
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
    )

    print("\nTop Ridge Coefficients:")
    print(coef_df.head(10).to_string(index=False))

    run_id = f"ridge_bench_{cfg.market}_{cfg.target_name}_h{cfg.horizon_days}d_{_utc_run_ts()}"
    out_dir = cfg.artifacts_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out_dir / "metrics_comparison.csv", index=False)
    coef_df.to_csv(out_dir / "coefficients.csv", index=False)

    preds = test[["as_of_date"]].copy()
    preds["y_true"] = y_test
    preds["y_pred_ridge"] = pred_ridge_test
    preds["y_pred_zero"] = pred_zero_test
    preds["y_pred_train_mean"] = pred_mean_test
    preds.to_csv(out_dir / "preds_test_comparison.csv", index=False)

    meta = {
        "market": cfg.market,
        "target_name": cfg.target_name,
        "horizon_days": cfg.horizon_days,
        "feature_version": cfg.feature_version,
        "target_version": cfg.target_version,
        "alpha": cfg.alpha,
        "test_window_days": cfg.test_window_days,
        "train_mean_target": mu,
        "rows_total": len(df),
        "rows_train": len(train),
        "rows_test": len(test),
    }
    pd.Series(meta).to_csv(out_dir / "run_meta.csv")

    print(f"\nArtifacts saved to: {out_dir}")


def run_ridge_for_markets(
    markets: list[str],
    *,
    target_name: str = "fwd_ret_5d_log",
    horizon_days: int = 5,
    feature_version: str = "v2",
    target_version: str = "v2",
    alpha: float = 1.0,
) -> None:
    for m in markets:
        cfg = RidgeConfig(
            market=m,
            target_name=target_name,
            horizon_days=horizon_days,
            feature_version=feature_version,
            target_version=target_version,
            alpha=alpha,
        )
        run_step5_ridge_with_step55_benchmarks(cfg)


if __name__ == "__main__":
    cfg = RidgeConfig()
    run_step5_ridge_with_step55_benchmarks(cfg)

    # Run:
    #   python -m src.models.common.baselines
    #   python -c "from src.models.common.baselines import run_ridge_for_markets; run_ridge_for_markets(['SPY','QQQ'])"
