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
from components.data_loader import load_signal_frames  # noqa: E402


def _fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):.1%}"


def _fmt_num(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):.1f}"


st.title("Forecast Console")
st.caption("Decision-oriented forecast view across horizons, model gates, and challenger outcomes.")

state, signals, history, model_health, backtests = load_signal_frames()
summary = state.get("summary", {})

if signals.empty:
    st.warning("No signal data found. Run `python main.py --skip-ui` first.")
    st.stop()

signals = signals.copy()
signals["event_probability"] = pd.to_numeric(signals["event_probability"], errors="coerce")
signals["horizon_days"] = pd.to_numeric(signals["horizon_days"], errors="coerce")
signals["model_health_score"] = pd.to_numeric(signals["model_health_score"], errors="coerce")
signals["strategy_edge_score"] = pd.to_numeric(signals.get("strategy_edge_score"), errors="coerce")
if "challenger_winner" not in signals.columns:
    signals["challenger_winner"] = "Not Run"
if "freshness_status" not in signals.columns:
    signals["freshness_status"] = summary.get("freshness_status", "Unknown")

left, right = st.columns([1, 1])
with left:
    horizon_filter = st.multiselect(
        "Horizons",
        sorted(signals["horizon_days"].dropna().astype(int).unique().tolist()),
        default=sorted(signals["horizon_days"].dropna().astype(int).unique().tolist()),
    )
with right:
    gate_filter = st.multiselect(
        "Model Gates",
        sorted(signals["model_gate"].dropna().unique().tolist()),
        default=sorted(signals["model_gate"].dropna().unique().tolist()),
    )

if horizon_filter:
    signals = signals[signals["horizon_days"].isin(horizon_filter)].copy()
if gate_filter:
    signals = signals[signals["model_gate"].isin(gate_filter)].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Consensus Action", summary.get("consensus_action", "n/a"))
c2.metric("Consensus Probability", _fmt_pct(summary.get("consensus_probability")))
c3.metric("Approved Signals", int(summary.get("approved_signal_count", 0)))
c4.metric("XGBoost Wins", int(summary.get("xgb_win_count", 0)))

st.subheader("Forecast Ladder")
ladder = signals[
    [
        "event_rule",
        "horizon_days",
        "event_probability",
        "recommendation",
        "model_gate",
        "decision_use",
        "strategy_edge_score",
        "challenger_winner",
        "freshness_status",
    ]
].copy()
ladder = ladder.sort_values(
    ["model_gate", "strategy_edge_score", "event_probability"],
    ascending=[True, False, False],
)
ladder["event_probability"] = ladder["event_probability"].map(_fmt_pct)
ladder["strategy_edge_score"] = ladder["strategy_edge_score"].map(_fmt_num)
ladder = ladder.rename(
    columns={
        "event_rule": "Rule",
        "horizon_days": "Horizon (days)",
        "event_probability": "Probability",
        "recommendation": "Risk Posture",
        "model_gate": "Model Gate",
        "decision_use": "Use",
        "strategy_edge_score": "Edge Score",
        "challenger_winner": "Challenger Winner",
        "freshness_status": "Freshness",
    }
)
st.dataframe(ladder, use_container_width=True, hide_index=True)

st.subheader("Model Comparison")
if model_health.empty:
    st.info("No model health data available.")
else:
    mh = model_health.copy()
    if "challenger_winner" not in mh.columns:
        mh["challenger_winner"] = "Not Run"
    for col in ["evaluation_auc", "xgb_auc", "evaluation_ap", "xgb_ap"]:
        if col not in mh.columns:
            mh[col] = pd.NA
    mh["evaluation_auc"] = pd.to_numeric(mh.get("evaluation_auc"), errors="coerce")
    mh["xgb_auc"] = pd.to_numeric(mh.get("xgb_auc"), errors="coerce")
    mh["evaluation_ap"] = pd.to_numeric(mh.get("evaluation_ap"), errors="coerce")
    mh["xgb_ap"] = pd.to_numeric(mh.get("xgb_ap"), errors="coerce")
    compare = mh[
        [
            "event_rule",
            "horizon_days",
            "model_gate",
            "challenger_winner",
            "evaluation_auc",
            "xgb_auc",
            "evaluation_ap",
            "xgb_ap",
        ]
    ].copy()
    for col in ["evaluation_auc", "xgb_auc", "evaluation_ap", "xgb_ap"]:
        compare[col] = compare[col].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "n/a")
    compare = compare.rename(
        columns={
            "event_rule": "Rule",
            "horizon_days": "Horizon (days)",
            "model_gate": "Gate",
            "challenger_winner": "Winner",
            "evaluation_auc": "Logistic AUC",
            "xgb_auc": "XGBoost AUC",
            "evaluation_ap": "Logistic AP",
            "xgb_ap": "XGBoost AP",
        }
    )
    st.dataframe(compare, use_container_width=True, hide_index=True)

render_ai_explainer(
    page_key="forecast",
    state=state,
    signals=signals,
    history=history,
    model_health=model_health,
    backtests=backtests,
)
