from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.models.common.risk_event_logistic import (
    EventLogitConfig,
    _auto_tune_defaults,
    _fit_event_definition,
    _make_event_labels,
    _select_feature_columns,
    _split_fit_calib_test,
    _topk_alerts,
)
from src.training.training_matrix import build_training_matrix_asof


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _metric_row(tag: str, y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    has_two = len(np.unique(y)) > 1
    return {
        "tag": tag,
        "event_rate": float(y.mean()) if len(y) else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if has_two else float("nan"),
        "ap": float(average_precision_score(y, p)) if has_two else float("nan"),
        "p_mean": float(p.mean()),
        "p_p95": float(np.quantile(p, 0.95)),
        "p_p99": float(np.quantile(p, 0.99)),
    }


def _driver_rows(
    *,
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    if X.empty:
        return pd.DataFrame(columns=["feature", "contribution"])

    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(X[feature_cols].values, feature_names=feature_cols), pred_contribs=True)
    latest = contribs[-1]
    rows = []
    for feature, value in zip(feature_cols, latest[:-1]):
        rows.append({"feature": feature, "contribution": float(value)})
    out = pd.DataFrame(rows)
    out["abs_contribution"] = out["contribution"].abs()
    return out.sort_values("abs_contribution", ascending=False).drop(columns=["abs_contribution"]).head(12)


def run_step6_event_xgboost(cfg: EventLogitConfig) -> Path:
    cfg = _auto_tune_defaults(cfg)

    if cfg.as_of_ts is None:
        cfg.as_of_ts = _utc_now()
    elif cfg.as_of_ts.tzinfo is None:
        cfg.as_of_ts = cfg.as_of_ts.replace(tzinfo=timezone.utc)
    else:
        cfg.as_of_ts = cfg.as_of_ts.astimezone(timezone.utc)

    print("\n=== STEP 6.X: XGBoost Risk-Event Challenger ===")
    print(
        "Config: "
        f"market={cfg.market} target={cfg.target_name} h={cfg.horizon_days} "
        f"rule={cfg.event_rule} as_of_ts={cfg.as_of_ts.isoformat()}"
    )

    df = build_training_matrix_asof(
        market=cfg.market,
        as_of_ts=cfg.as_of_ts,
        feature_version=cfg.feature_version,
        target_name=cfg.target_name,
        horizon_days=cfg.horizon_days,
        target_version=cfg.target_version,
        dropna_features=True,
    )
    if df.empty:
        raise ValueError("Training matrix is empty after joins/dropna.")

    fit_df, calib_df, test_df = _split_fit_calib_test(
        df,
        test_n=int(cfg.test_window_days),
        calib_n=int(cfg.calib_window_days),
        mode=cfg.calib_split,
        fit_max_rows=cfg.fit_max_rows,
        fit_tail_only=cfg.fit_tail_only,
        debug_splits=cfg.debug_splits,
    )
    print(f"Rows total: {len(df)} | Fit: {len(fit_df)} | Calib: {len(calib_df)} | Test: {len(test_df)}")

    evdef = _fit_event_definition(fit_df, cfg=cfg)
    y_fit, _ = _make_event_labels(fit_df, cfg=cfg, evdef=evdef)
    y_calib, _ = _make_event_labels(calib_df, cfg=cfg, evdef=evdef)
    y_test, _ = _make_event_labels(test_df, cfg=cfg, evdef=evdef)

    feature_cols = _select_feature_columns(df)
    X_fit = fit_df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    X_calib = calib_df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)

    pos = max(1, int(y_fit.sum()))
    neg = max(1, int(len(y_fit) - y_fit.sum()))
    scale_pos_weight = neg / pos

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3 if cfg.horizon_days <= 21 else 4,
        min_child_weight=4,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=2,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_fit, y_fit)

    p_fit = model.predict_proba(X_fit)[:, 1]
    p_calib = model.predict_proba(X_calib)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    metrics_df = pd.DataFrame(
        [
            _metric_row("fit_raw", y_fit, p_fit),
            _metric_row("calib_raw", y_calib, p_calib),
            _metric_row("test_raw", y_test, p_test),
        ]
    )
    print("\n--- XGBoost Probability Metrics ---")
    print(metrics_df.to_string(index=False))

    topk_df = pd.DataFrame(_topk_alerts("xgb_prob", y_test, p_test))
    if not topk_df.empty:
        print("\n--- XGBoost Top-K Alert Summary ---")
        print(topk_df.to_string(index=False))

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    drivers_df = _driver_rows(model=model, X=X_test, feature_cols=feature_cols)

    stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    rule_tag = f"q{int(round(cfg.event_quantile * 100)):02d}" if cfg.event_rule == "quantile" else f"s{cfg.sigma_mult:.2f}"
    tag = f"eventxgb_{cfg.event_rule}_{rule_tag}_topk_{cfg.market}_{cfg.target_name}_h{cfg.horizon_days}d_{stamp}"
    out_dir = Path(cfg.out_dir) / tag
    _ensure_dir(out_dir)

    run_meta = {
        "market": cfg.market,
        "target_name": cfg.target_name,
        "horizon_days": cfg.horizon_days,
        "event_rule": cfg.event_rule,
        "feature_version": cfg.feature_version,
        "target_version": cfg.target_version,
        "model_family": "xgboost",
        "as_of_ts": cfg.as_of_ts.isoformat(),
    }
    pd.DataFrame([run_meta]).to_csv(out_dir / "run_meta.csv", index=False)
    metrics_df.to_csv(out_dir / "test_metrics.csv", index=False)
    topk_df.to_csv(out_dir / "topk_alerts.csv", index=False)
    importance_df.to_csv(out_dir / "feature_importances.csv", index=False)
    drivers_df.to_csv(out_dir / "latest_feature_drivers.csv", index=False)

    scored = test_df[[c for c in ["market", "as_of_date", "forward_date", "target_value"] if c in test_df.columns]].copy()
    scored["y_event"] = y_test
    scored["p_raw"] = p_test
    scored.to_csv(out_dir / "test_scored.csv", index=False)

    y_full, _ = _make_event_labels(df, cfg=cfg, evdef=evdef)
    X_full = df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    p_full = model.predict_proba(X_full)[:, 1]
    full_scored = df[
        [c for c in ["market", "as_of_date", "forward_date", "target_value"] if c in df.columns]
    ].copy()
    full_scored["y_event"] = y_full
    full_scored["p_raw"] = p_full
    full_scored.to_csv(out_dir / "full_scored.csv", index=False)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print("\nTop XGBoost Features:")
    print(importance_df.head(12).to_string(index=False))
    if not drivers_df.empty:
        print("\nLatest XGBoost Feature Drivers:")
        print(drivers_df.to_string(index=False))
    print(f"\nXGBoost artifacts saved to: {out_dir.as_posix()}")
    return out_dir
