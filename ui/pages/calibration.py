from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.ai_explainer import render_ai_explainer  # noqa: E402
from components.chart_utils import build_metric_rank_chart  # noqa: E402
from components.data_loader import load_signal_frames  # noqa: E402


st.title("Calibration & Model Health")
st.caption("Reliability-focused view across logistic metrics, backtests, and the XGBoost challenger.")

state, signals, history, model_health, backtests = load_signal_frames()
if model_health.empty:
    st.warning("No model health data found. Run `python main.py --skip-ui` first.")
    st.stop()

mh = model_health.copy()
for col in [
    "model_health_score",
    "evaluation_brier",
    "evaluation_auc",
    "evaluation_ap",
    "evaluation_drift_gap_abs_mean",
    "mean_test_brier_prod",
    "mean_test_auc_prod",
    "mean_test_ap_prod",
    "xgb_brier",
    "xgb_auc",
    "xgb_ap",
]:
    if col not in mh.columns:
        mh[col] = pd.NA
    mh[col] = pd.to_numeric(mh[col], errors="coerce")
if "challenger_winner" not in mh.columns:
    mh["challenger_winner"] = "Not Run"

metric = st.selectbox(
    "Metric",
    ["evaluation_auc", "evaluation_brier", "mean_test_auc_prod", "mean_test_brier_prod", "xgb_auc", "xgb_brier"],
    index=0,
)

plot_df = mh[["event_rule", "horizon_days", metric, "model_gate"]].dropna(subset=[metric]).copy()
plot_df["label"] = plot_df["event_rule"].astype(str) + " | " + plot_df["horizon_days"].astype(str) + "d"

if plot_df.empty:
    st.info("No data available for the selected metric.")
else:
    st.caption(
        "This ranking view is easier to scan than a crowded categorical bar chart. Higher is better for AUC and lower is better for Brier."
    )
    fig = build_metric_rank_chart(
        plot_df,
        metric=metric,
        title=f"{metric.replace('_', ' ').title()} by Signal",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Calibration Table")
table = mh[
    [
        "event_rule",
        "horizon_days",
        "model_gate",
        "model_health_label",
        "model_health_score",
        "evaluation_brier",
        "evaluation_auc",
        "evaluation_ap",
        "evaluation_drift_gap_abs_mean",
        "xgb_brier",
        "xgb_auc",
        "xgb_ap",
        "challenger_winner",
    ]
].copy()
for col in [
    "model_health_score",
    "evaluation_brier",
    "evaluation_auc",
    "evaluation_ap",
    "evaluation_drift_gap_abs_mean",
    "xgb_brier",
    "xgb_auc",
    "xgb_ap",
]:
    table[col] = table[col].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "n/a")
table = table.rename(
    columns={
        "event_rule": "Rule",
        "horizon_days": "Horizon (days)",
        "model_gate": "Gate",
        "model_health_label": "Health",
        "model_health_score": "Health Score",
        "evaluation_brier": "Logistic Brier",
        "evaluation_auc": "Logistic AUC",
        "evaluation_ap": "Logistic AP",
        "evaluation_drift_gap_abs_mean": "Mean |Drift Gap|",
        "xgb_brier": "XGBoost Brier",
        "xgb_auc": "XGBoost AUC",
        "xgb_ap": "XGBoost AP",
        "challenger_winner": "Winner",
    }
)
st.dataframe(table, use_container_width=True, hide_index=True)

render_ai_explainer(
    page_key="calibration",
    state=state,
    signals=signals,
    history=history,
    model_health=model_health,
    backtests=backtests,
)
