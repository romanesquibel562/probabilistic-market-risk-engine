# src/reporting/product_summary.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.reporting.model_evaluator import (
    build_backtest_eval_summary,
    build_model_eval_summary,
    build_xgb_eval_summary,
)


@dataclass(frozen=True)
class ProductSummaryPaths:
    signals_csv: Path
    history_csv: Path
    model_health_csv: Path
    state_json: Path


def _repo_root(repo_root: Path | None = None) -> Path:
    return repo_root or Path(__file__).resolve().parents[2]


def _outputs_dir(repo_root: Path | None = None) -> Path:
    out = _repo_root(repo_root) / "artifacts" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_friendly_history(market: str, repo_root: Path | None = None) -> pd.DataFrame:
    outputs = _outputs_dir(repo_root)
    files = sorted(outputs.glob(f"friendly_risk_summary_{market}_*.csv"))
    frames: list[pd.DataFrame] = []

    for path in files:
        df = _safe_read_csv(path)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    history = pd.concat(frames, ignore_index=True)
    history["as_of_date"] = pd.to_datetime(history["as_of_date"], errors="coerce")
    history["horizon_days"] = history["horizon"].astype(str).str.extract(r"(\d+)").iloc[:, 0]
    history["horizon_days"] = pd.to_numeric(history["horizon_days"], errors="coerce")
    history["event_probability"] = pd.to_numeric(history["event_probability"], errors="coerce")
    history["event_rate_rolling"] = pd.to_numeric(history["event_rate_rolling"], errors="coerce")
    history["calibration_gap"] = pd.to_numeric(history["calibration_gap"], errors="coerce")
    history = history.dropna(
        subset=["as_of_date", "event_rule", "horizon_days", "event_probability"]
    ).copy()
    history["horizon_days"] = history["horizon_days"].astype(int)
    history = history.sort_values(
        ["event_rule", "horizon_days", "as_of_date", "source_file"]
    ).drop_duplicates(
        subset=["market", "event_rule", "horizon_days", "as_of_date"], keep="last"
    )
    return history.reset_index(drop=True)


def _percentile_rank(values: pd.Series) -> list[float]:
    pct: list[float] = []
    observed: list[float] = []
    for val in values.astype(float).tolist():
        observed.append(float(val))
        pct.append(float(sum(x <= val for x in observed) / len(observed)))
    return pct


def _classify_trend(delta_short: float, delta_medium: float) -> str:
    if pd.isna(delta_short) and pd.isna(delta_medium):
        return "stable"
    if (delta_short or 0.0) >= 0.02 or (delta_medium or 0.0) >= 0.04:
        return "rising"
    if (delta_short or 0.0) <= -0.02 or (delta_medium or 0.0) <= -0.04:
        return "falling"
    return "stable"


def _recommendation(probability: float, percentile: float, health_score: float) -> str:
    if pd.isna(probability):
        return "Unavailable"
    if health_score < 45:
        return "Hold / Low Confidence"
    if probability >= 0.30 or percentile >= 0.90:
        return "High Alert"
    if probability >= 0.20 or percentile >= 0.75:
        return "Defensive"
    if probability >= 0.12 or percentile >= 0.60:
        return "Caution"
    return "Normal"


def _health_label(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 85:
        return "Strong"
    if score >= 70:
        return "Healthy"
    if score >= 55:
        return "Watch"
    return "Fragile"


def _freshness_status(age_days: int | float | None) -> str:
    if age_days is None or pd.isna(age_days):
        return "Unknown"
    age_days = int(age_days)
    if age_days <= 3:
        return "Fresh"
    if age_days <= 7:
        return "Aging"
    return "Stale"


def _model_gate(
    *,
    health_score: float | None,
    eval_auc: float | None,
    topk_precision: float | None,
    backtest_auc: float | None,
) -> str:
    hs = np.nan if health_score is None else float(health_score)
    ea = np.nan if eval_auc is None else float(eval_auc)
    tp = np.nan if topk_precision is None else float(topk_precision)
    ba = np.nan if backtest_auc is None else float(backtest_auc)

    if not np.isnan(hs) and hs >= 70 and ((not np.isnan(ea) and ea >= 0.53) or (not np.isnan(ba) and ba >= 0.53)) and ((np.isnan(tp)) or tp >= 0.12):
        return "Approved"
    if not np.isnan(hs) and hs >= 55 and ((np.isnan(ea)) or ea >= 0.51):
        return "Watch"
    if not np.isnan(hs) and hs >= 40:
        return "Research Only"
    return "Suppressed"


def _consensus_action(probability: float, approved_ratio: float, freshness: str) -> str:
    if freshness == "Stale":
        return "Cached / Review Freshness"
    if probability >= 0.28 and approved_ratio >= 0.4:
        return "De-Risk Aggressively"
    if probability >= 0.20:
        return "Reduce Exposure"
    if probability >= 0.12:
        return "Monitor Closely"
    return "Normal Risk Posture"


def _agreement_label(spread: float) -> str:
    if spread <= 0.04:
        return "High Agreement"
    if spread <= 0.10:
        return "Mixed"
    return "Low Agreement"


def _strategy_edge_score(brier: float | None, auc: float | None, ap: float | None) -> float:
    b = np.nan if brier is None else float(brier)
    a = np.nan if auc is None else float(auc)
    p = np.nan if ap is None else float(ap)

    brier_part = max(0.0, 0.25 - b) * 300 if not np.isnan(b) else 0.0
    auc_part = max(0.0, a - 0.50) * 200 if not np.isnan(a) else 0.0
    ap_part = max(0.0, p - 0.10) * 120 if not np.isnan(p) else 0.0
    return round(float(brier_part + auc_part + ap_part), 1)


def _challenger_winner(
    logistic_auc: float | None,
    logistic_brier: float | None,
    xgb_auc: float | None,
    xgb_brier: float | None,
) -> str:
    la = np.nan if logistic_auc is None else float(logistic_auc)
    lb = np.nan if logistic_brier is None else float(logistic_brier)
    xa = np.nan if xgb_auc is None else float(xgb_auc)
    xb = np.nan if xgb_brier is None else float(xgb_brier)

    if np.isnan(xa):
        return "Logistic Only"
    if np.isnan(la):
        return "XGBoost"
    if xa >= la + 0.01 and (np.isnan(lb) or np.isnan(xb) or xb <= lb + 0.01):
        return "XGBoost"
    if la >= xa + 0.01 and (np.isnan(lb) or np.isnan(xb) or lb <= xb + 0.01):
        return "Logistic"
    return "Mixed"


def _compute_health_score(
    brier: float | None,
    auc: float | None,
    ap: float | None,
    abs_gap: float | None,
) -> float:
    brier_v = np.nan if brier is None else float(brier)
    auc_v = np.nan if auc is None else float(auc)
    ap_v = np.nan if ap is None else float(ap)
    gap_v = np.nan if abs_gap is None else float(abs_gap)

    brier_component = max(0.0, 1.0 - (brier_v / 0.25)) * 40 if not np.isnan(brier_v) else 0.0
    auc_component = np.clip((auc_v - 0.50) / 0.15, 0.0, 1.0) * 25 if not np.isnan(auc_v) else 0.0
    ap_component = np.clip(ap_v / 0.40, 0.0, 1.0) * 15 if not np.isnan(ap_v) else 0.0
    gap_component = max(0.0, 1.0 - (gap_v / 0.20)) * 20 if not np.isnan(gap_v) else 0.0
    return round(float(brier_component + auc_component + ap_component + gap_component), 1)


def build_dashboard_state(
    market: str = "SPY",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    history = _load_friendly_history(market=market, repo_root=repo_root)
    eval_df = build_model_eval_summary(market=market, repo_root=repo_root, latest_only=True)
    xgb_df = build_xgb_eval_summary(market=market, repo_root=repo_root, latest_only=True)
    backtest_df = build_backtest_eval_summary(market=market, repo_root=repo_root, latest_only=True)

    if history.empty:
        return {
            "market": market,
            "latest_as_of_date": None,
            "signals": [],
            "signal_history": [],
            "model_health": [],
            "backtests": [],
            "summary": {},
        }

    derived_frames: list[pd.DataFrame] = []
    group_cols = ["market", "event_rule", "horizon_days"]

    for _, grp in history.groupby(group_cols, sort=False):
        g = grp.sort_values("as_of_date").copy()
        g["probability_delta_1obs"] = g["event_probability"].diff(1)
        g["probability_delta_5obs"] = g["event_probability"].diff(5)
        g["probability_mean_5obs"] = g["event_probability"].rolling(5, min_periods=2).mean()
        g["probability_std_5obs"] = g["event_probability"].rolling(5, min_periods=2).std()
        g["probability_min_20obs"] = g["event_probability"].rolling(20, min_periods=2).min()
        g["probability_max_20obs"] = g["event_probability"].rolling(20, min_periods=2).max()
        g["historical_percentile"] = _percentile_rank(g["event_probability"])
        g["signal_zscore"] = (
            (g["event_probability"] - g["probability_mean_5obs"]) / g["probability_std_5obs"]
        )
        g["signal_trend"] = [
            _classify_trend(short, medium)
            for short, medium in zip(
                g["probability_delta_1obs"],
                g["probability_delta_5obs"],
            )
        ]
        derived_frames.append(g)

    signal_history = pd.concat(derived_frames, ignore_index=True)
    latest_signals = (
        signal_history.sort_values("as_of_date")
        .groupby(group_cols, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    eval_subset = pd.DataFrame()
    if not eval_df.empty:
        eval_subset = eval_df.copy()
        eval_subset["event_rule"] = eval_subset["event_rule"].fillna("unknown")
        eval_subset["horizon_days"] = pd.to_numeric(
            eval_subset["horizon_days"], errors="coerce"
        ).fillna(-1).astype(int)
        eval_subset = eval_subset[
            [
                "event_rule",
                "horizon_days",
                "artifact_dir",
                "m_brier",
                "m_auc",
                "m_ap",
                "topk_precision_last",
                "topk_recall_last",
                "topk_f1_last",
                "drift_gap_abs_mean",
                "drift_gap_last",
            ]
        ].rename(
            columns={
                "artifact_dir": "eval_artifact_dir",
                "m_brier": "evaluation_brier",
                "m_auc": "evaluation_auc",
                "m_ap": "evaluation_ap",
                "topk_precision_last": "evaluation_topk_precision",
                "topk_recall_last": "evaluation_topk_recall",
                "topk_f1_last": "evaluation_topk_f1",
                "drift_gap_abs_mean": "evaluation_drift_gap_abs_mean",
                "drift_gap_last": "evaluation_drift_gap_last",
            }
        )

    backtest_subset = pd.DataFrame()
    if not backtest_df.empty:
        backtest_subset = backtest_df.copy()
        backtest_subset["horizon_days"] = pd.to_numeric(
            backtest_subset["horizon_days"], errors="coerce"
        ).fillna(-1).astype(int)
        backtest_subset = backtest_subset.rename(
            columns={
                "artifact_dir": "backtest_artifact_dir",
            }
        )

    xgb_subset = pd.DataFrame()
    if not xgb_df.empty:
        xgb_subset = xgb_df.copy()
        xgb_subset["event_rule"] = xgb_subset["event_rule"].fillna("unknown")
        xgb_subset["horizon_days"] = pd.to_numeric(
            xgb_subset["horizon_days"], errors="coerce"
        ).fillna(-1).astype(int)
        xgb_subset = xgb_subset[
            [
                "event_rule",
                "horizon_days",
                "artifact_dir",
                "m_brier",
                "m_auc",
                "m_ap",
                "topk_precision_last",
                "topk_recall_last",
                "topk_f1_last",
            ]
        ].rename(
            columns={
                "artifact_dir": "xgb_artifact_dir",
                "m_brier": "xgb_brier",
                "m_auc": "xgb_auc",
                "m_ap": "xgb_ap",
                "topk_precision_last": "xgb_topk_precision",
                "topk_recall_last": "xgb_topk_recall",
                "topk_f1_last": "xgb_topk_f1",
            }
        )

    latest_signals = latest_signals.merge(
        eval_subset,
        on=["event_rule", "horizon_days"],
        how="left",
    )
    latest_signals = latest_signals.merge(
        backtest_subset,
        on=["market", "horizon_days"],
        how="left",
    )
    latest_signals = latest_signals.merge(
        xgb_subset,
        on=["event_rule", "horizon_days"],
        how="left",
    )

    latest_signals["calibration_gap_abs"] = latest_signals["calibration_gap"].abs()
    latest_signals["model_health_score"] = [
        _compute_health_score(brier, auc, ap, gap)
        for brier, auc, ap, gap in zip(
            latest_signals.get("evaluation_brier", pd.Series(dtype=float)),
            latest_signals.get("evaluation_auc", pd.Series(dtype=float)),
            latest_signals.get("evaluation_ap", pd.Series(dtype=float)),
            latest_signals.get("evaluation_drift_gap_abs_mean", pd.Series(dtype=float)),
        )
    ]
    latest_signals["model_health_label"] = latest_signals["model_health_score"].apply(_health_label)
    latest_signals["strategy_edge_score"] = [
        _strategy_edge_score(brier, auc, ap)
        for brier, auc, ap in zip(
            latest_signals.get("mean_test_brier_prod", pd.Series(dtype=float)),
            latest_signals.get("mean_test_auc_prod", pd.Series(dtype=float)),
            latest_signals.get("mean_test_ap_prod", pd.Series(dtype=float)),
        )
    ]
    latest_signals["model_gate"] = [
        _model_gate(
            health_score=hs,
            eval_auc=ea,
            topk_precision=tp,
            backtest_auc=ba,
        )
        for hs, ea, tp, ba in zip(
            latest_signals["model_health_score"],
            latest_signals.get("evaluation_auc", pd.Series(dtype=float)),
            latest_signals.get("evaluation_topk_precision", pd.Series(dtype=float)),
            latest_signals.get("mean_test_auc_prod", pd.Series(dtype=float)),
        )
    ]
    latest_signals["recommendation"] = [
        _recommendation(prob, pct, score)
        for prob, pct, score in zip(
            latest_signals["event_probability"],
            latest_signals["historical_percentile"],
            latest_signals["model_health_score"],
        )
    ]

    latest_signals["as_of_date"] = pd.to_datetime(
        latest_signals["as_of_date"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    today_utc = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

    latest_signals["days_since_latest_signal"] = (
        today_utc - latest_signals["as_of_date"]
    ).dt.days
    latest_signals["freshness_status"] = latest_signals["days_since_latest_signal"].apply(_freshness_status)
    latest_signals["decision_use"] = np.where(
        latest_signals["model_gate"].isin(["Approved", "Watch"]),
        "Decision Candidate",
        "Research Only",
    )
    latest_signals["challenger_winner"] = [
        _challenger_winner(la, lb, xa, xb)
        for la, lb, xa, xb in zip(
            latest_signals.get("evaluation_auc", pd.Series(dtype=float)),
            latest_signals.get("evaluation_brier", pd.Series(dtype=float)),
            latest_signals.get("xgb_auc", pd.Series(dtype=float)),
            latest_signals.get("xgb_brier", pd.Series(dtype=float)),
        )
    ]
    latest_signals["xgb_available"] = latest_signals.get("xgb_auc", pd.Series(dtype=float)).notna()

    latest_signals = latest_signals.sort_values(
        ["horizon_days", "event_rule"]
    ).reset_index(drop=True)

    approved = latest_signals[latest_signals["model_gate"].isin(["Approved", "Watch"])].copy()
    if approved.empty:
        approved = latest_signals.copy()

    weights = approved["model_health_score"].clip(lower=1.0)
    consensus_probability = float(np.average(approved["event_probability"], weights=weights))
    approved_ratio = float(
        latest_signals["model_gate"].isin(["Approved", "Watch"]).mean()
    )
    signal_spread = float(
        pd.to_numeric(latest_signals["event_probability"], errors="coerce").max()
        - pd.to_numeric(latest_signals["event_probability"], errors="coerce").min()
    )
    consensus = {
        "probability": round(consensus_probability, 4),
        "action": _consensus_action(
            consensus_probability,
            approved_ratio,
            _freshness_status(latest_signals["days_since_latest_signal"].min()),
        ),
        "approved_ratio": round(approved_ratio, 3),
        "signal_spread": round(signal_spread, 4),
        "agreement": _agreement_label(signal_spread),
    }

    model_health = latest_signals[
        [
            "market",
            "event_rule",
            "horizon_days",
            "model_health_score",
            "model_health_label",
            "model_gate",
            "decision_use",
            "strategy_edge_score",
            "challenger_winner",
            "xgb_available",
            "evaluation_brier",
            "evaluation_auc",
            "evaluation_ap",
            "evaluation_drift_gap_abs_mean",
            "evaluation_topk_precision",
            "evaluation_topk_recall",
            "evaluation_topk_f1",
            "mean_test_brier_prod",
            "mean_test_auc_prod",
            "mean_test_ap_prod",
            "chosen_stream_mode",
            "xgb_brier",
            "xgb_auc",
            "xgb_ap",
            "xgb_topk_precision",
            "xgb_topk_recall",
            "xgb_topk_f1",
        ]
    ].copy()

    latest_date = latest_signals["as_of_date"].max()
    freshness_age_days = int(latest_signals["days_since_latest_signal"].min()) if not latest_signals.empty else None
    summary = {
        "market": market,
        "latest_as_of_date": latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else None,
        "freshness_age_days": freshness_age_days,
        "freshness_status": _freshness_status(freshness_age_days),
        "signal_count": int(len(latest_signals)),
        "average_health_score": round(float(latest_signals["model_health_score"].mean()), 1),
        "approved_signal_count": int(latest_signals["model_gate"].isin(["Approved", "Watch"]).sum()),
        "suppressed_signal_count": int((latest_signals["model_gate"] == "Suppressed").sum()),
        "elevated_signal_count": int(
            latest_signals["recommendation"].isin(["High Alert", "Defensive"]).sum()
        ),
        "high_alert_count": int((latest_signals["recommendation"] == "High Alert").sum()),
        "average_probability": round(float(latest_signals["event_probability"].mean()), 4),
        "consensus_probability": consensus["probability"],
        "consensus_action": consensus["action"],
        "signal_agreement": consensus["agreement"],
        "xgb_available_count": int(latest_signals["xgb_available"].sum()),
        "xgb_win_count": int((latest_signals["challenger_winner"] == "XGBoost").sum()),
        "best_xgb_auc": round(
            float(pd.to_numeric(xgb_df.get("m_auc"), errors="coerce").max()),
            4,
        )
        if not xgb_df.empty
        else None,
        "best_backtest_auc": round(
            float(pd.to_numeric(backtest_df.get("mean_test_auc_prod"), errors="coerce").max()),
            4,
        )
        if not backtest_df.empty
        else None,
    }

    return {
        "market": market,
        "latest_as_of_date": summary["latest_as_of_date"],
        "signals": latest_signals.assign(
            as_of_date=lambda d: d["as_of_date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
        "signal_history": signal_history.assign(
            as_of_date=lambda d: d["as_of_date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
        "model_health": model_health.to_dict(orient="records"),
        "backtests": backtest_df.to_dict(orient="records"),
        "consensus": consensus,
        "summary": summary,
    }


def write_product_outputs(
    market: str = "SPY",
    as_of_date: str = "latest",
    repo_root: Path | None = None,
) -> ProductSummaryPaths:
    state = build_dashboard_state(market=market, repo_root=repo_root)
    outputs = _outputs_dir(repo_root)

    signals_path = outputs / f"dashboard_signals_{market}_{as_of_date}.csv"
    history_path = outputs / f"dashboard_signal_history_{market}_{as_of_date}.csv"
    health_path = outputs / f"dashboard_model_health_{market}_{as_of_date}.csv"
    state_path = outputs / f"dashboard_state_{market}_{as_of_date}.json"

    pd.DataFrame(state["signals"]).to_csv(signals_path, index=False)
    pd.DataFrame(state["signal_history"]).to_csv(history_path, index=False)
    pd.DataFrame(state["model_health"]).to_csv(health_path, index=False)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    return ProductSummaryPaths(
        signals_csv=signals_path,
        history_csv=history_path,
        model_health_csv=health_path,
        state_json=state_path,
    )


if __name__ == "__main__":
    paths = write_product_outputs(market="SPY", as_of_date="latest")
    print("[product_summary] wrote:", paths.signals_csv)
    print("[product_summary] wrote:", paths.history_csv)
    print("[product_summary] wrote:", paths.model_health_csv)
    print("[product_summary] wrote:", paths.state_json)
