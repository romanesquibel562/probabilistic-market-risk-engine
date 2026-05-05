from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.data_loader import load_signal_frames, load_spy_prices  # noqa: E402


st.set_page_config(
    page_title="Market Risk Engine",
    page_icon="MR",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(248, 210, 157, 0.30), transparent 24%),
            radial-gradient(circle at top right, rgba(93, 144, 219, 0.20), transparent 26%),
            linear-gradient(180deg, #fbf7ef 0%, #f4efe5 100%);
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid rgba(81, 57, 38, 0.12);
        padding: 0.9rem 1rem;
        border-radius: 16px;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .risk-pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.92rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(81, 57, 38, 0.12);
        background: rgba(255, 255, 255, 0.72);
    }

    .section-card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(81, 57, 38, 0.10);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}%}"


def _format_num(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


st.sidebar.title("Market Risk Engine")
st.sidebar.caption("Decision-support dashboard for downside-risk monitoring")

market = st.sidebar.selectbox("Market", ["SPY"], index=0)
show_research_only = st.sidebar.toggle("Show research-only signals", value=True)
selected_horizons = st.sidebar.multiselect(
    "Horizons",
    [5, 21, 63],
    default=[5, 21, 63],
)

state, signals, history, model_health, backtests = load_signal_frames(market=market)
prices = load_spy_prices()
summary = state.get("summary", {})
consensus = state.get("consensus", {})

st.title("Market Risk Intelligence Dashboard")
st.caption(
    "A calibrated risk-monitoring system that separates decision candidates from weaker research signals."
)

if signals.empty:
    st.warning("No dashboard signals found yet. Run `python main.py --skip-ui` first.")
    st.stop()

signals = signals.copy()
signals["horizon_days"] = pd.to_numeric(signals["horizon_days"], errors="coerce")
signals["event_probability"] = pd.to_numeric(signals["event_probability"], errors="coerce")
signals["model_health_score"] = pd.to_numeric(signals["model_health_score"], errors="coerce")
signals["strategy_edge_score"] = pd.to_numeric(signals.get("strategy_edge_score"), errors="coerce")
if "challenger_winner" not in signals.columns:
    signals["challenger_winner"] = "Not Run"

if selected_horizons:
    signals = signals[signals["horizon_days"].isin(selected_horizons)].copy()
    history = history[history["horizon_days"].isin(selected_horizons)].copy()
    model_health = model_health[model_health["horizon_days"].isin(selected_horizons)].copy()
    backtests = backtests[backtests["horizon_days"].isin(selected_horizons)].copy()

if not show_research_only:
    signals = signals[signals["model_gate"].isin(["Approved", "Watch"])].copy()
    model_health = model_health[model_health["model_gate"].isin(["Approved", "Watch"])].copy()

if "challenger_winner" not in model_health.columns:
    model_health["challenger_winner"] = "Not Run"
for col in ["xgb_auc", "xgb_ap"]:
    if col not in model_health.columns:
        model_health[col] = np.nan

latest_date = pd.to_datetime(summary.get("latest_as_of_date"), errors="coerce")
freshness_status = summary.get("freshness_status", "Unknown")
freshness_age_days = summary.get("freshness_age_days")
consensus_action = summary.get("consensus_action", "n/a")

if freshness_status == "Stale":
    st.warning(
        f"Data freshness is `{freshness_status}`. Latest market date is "
        f"`{summary.get('latest_as_of_date', 'n/a')}` and the cache is about `{freshness_age_days}` days old."
    )
elif freshness_status == "Aging":
    st.info(
        f"Data freshness is `{freshness_status}`. Latest market date is "
        f"`{summary.get('latest_as_of_date', 'n/a')}`."
    )

st.markdown(
    f"""
    <div class="section-card">
      <div class="risk-pill"><strong>Consensus Action:</strong> {consensus_action}</div>
      <div class="risk-pill"><strong>Freshness:</strong> {freshness_status}</div>
      <div class="risk-pill"><strong>Agreement:</strong> {summary.get("signal_agreement", "n/a")}</div>
      <div class="risk-pill"><strong>Approved Signals:</strong> {summary.get("approved_signal_count", 0)}</div>
      <div class="risk-pill"><strong>Suppressed Signals:</strong> {summary.get("suppressed_signal_count", 0)}</div>
      <div class="risk-pill"><strong>XGBoost Wins:</strong> {summary.get("xgb_win_count", 0)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric(
    "Latest Signal Date",
    latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "n/a",
)
c2.metric("Consensus Probability", _format_pct(summary.get("consensus_probability"), digits=1))
c3.metric("Average Health Score", _format_num(summary.get("average_health_score"), digits=1))
c4.metric("Elevated Signals", int(summary.get("elevated_signal_count", 0)))
c5.metric("Best Backtest AUC", _format_num(summary.get("best_backtest_auc"), digits=3))
c6.metric("Best XGBoost AUC", _format_num(summary.get("best_xgb_auc"), digits=3))

board = signals[
    [
        "event_rule",
        "horizon_days",
        "event_probability",
        "probability_delta_1obs",
        "historical_percentile",
        "recommendation",
        "model_gate",
        "decision_use",
        "model_health_label",
        "model_health_score",
        "strategy_edge_score",
        "challenger_winner",
        "freshness_status",
    ]
].copy()

board["event_probability_num"] = pd.to_numeric(board["event_probability"], errors="coerce")
board["model_health_score_num"] = pd.to_numeric(board["model_health_score"], errors="coerce")
board["strategy_edge_score_num"] = pd.to_numeric(board["strategy_edge_score"], errors="coerce")

board["event_probability"] = board["event_probability_num"].map(_format_pct)
board["probability_delta_1obs"] = pd.to_numeric(
    board["probability_delta_1obs"], errors="coerce"
).map(lambda x: f"{x:+.1%}" if pd.notna(x) else "n/a")
board["historical_percentile"] = pd.to_numeric(
    board["historical_percentile"], errors="coerce"
).map(lambda x: f"{x:.0%}" if pd.notna(x) else "n/a")
board["model_health_score"] = board["model_health_score_num"].map(_format_num)
board["strategy_edge_score"] = board["strategy_edge_score_num"].map(_format_num)

board = board.sort_values(
    by=["model_gate", "strategy_edge_score_num", "event_probability_num"],
    ascending=[True, False, False],
).drop(columns=["event_probability_num", "model_health_score_num", "strategy_edge_score_num"])

board = board.rename(
    columns={
        "event_rule": "Rule",
        "horizon_days": "Horizon (days)",
        "event_probability": "Probability",
        "probability_delta_1obs": "1-Step Change",
        "historical_percentile": "Historical Percentile",
        "recommendation": "Risk Posture",
        "model_gate": "Model Gate",
        "decision_use": "Use",
        "model_health_label": "Health",
        "model_health_score": "Health Score",
        "strategy_edge_score": "Edge Score",
        "challenger_winner": "Challenger Winner",
        "freshness_status": "Freshness",
    }
)

st.subheader("Signal Decision Board")
st.dataframe(board, use_container_width=True, hide_index=True)

chart_left, chart_right = st.columns([1.35, 1])

with chart_left:
    st.subheader("Risk Signal History")
    if history.empty:
        st.info("No history available yet.")
    else:
        hist = history.copy()
        hist["series"] = hist["event_rule"].astype(str) + " | " + hist["horizon_days"].astype(str) + "d"
        fig = px.line(
            hist,
            x="as_of_date",
            y="event_probability",
            color="series",
            markers=True,
            labels={
                "as_of_date": "Date",
                "event_probability": "Probability",
                "series": "Signal",
            },
            title="Multi-Horizon Downside Probability",
        )
        fig.update_layout(
            legend_title_text="Signal",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

with chart_right:
    st.subheader("SPY Price Overlay")
    if prices.empty or history.empty:
        st.info("Price overlay unavailable.")
    else:
        overlay = history.copy()
        overlay = overlay.sort_values("as_of_date")
        sigma_21 = overlay[
            (overlay["event_rule"].astype(str) == "sigma")
            & (pd.to_numeric(overlay["horizon_days"], errors="coerce") == 21)
        ][["as_of_date", "event_probability"]].copy()
        sigma_21["event_probability"] = pd.to_numeric(sigma_21["event_probability"], errors="coerce")

        price_fig = go.Figure()
        price_fig.add_trace(
            go.Scatter(
                x=prices["date"],
                y=prices["close"],
                name="SPY Close",
                line=dict(color="#1f3b5d", width=2.2),
                yaxis="y1",
            )
        )
        if not sigma_21.empty:
            price_fig.add_trace(
                go.Scatter(
                    x=sigma_21["as_of_date"],
                    y=sigma_21["event_probability"],
                    name="21d Sigma Risk",
                    line=dict(color="#ad5c1f", width=2),
                    yaxis="y2",
                )
            )
        price_fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            yaxis=dict(title="SPY Close"),
            yaxis2=dict(
                title="Risk Probability",
                overlaying="y",
                side="right",
                tickformat=".0%",
            ),
        )
        st.plotly_chart(price_fig, use_container_width=True)

lower_left, lower_right = st.columns([1, 1])

with lower_left:
    st.subheader("Walk-Forward Backtest Snapshot")
    if backtests.empty:
        st.info("No backtest summary available yet.")
    else:
        bt = backtests[
            [
                "horizon_days",
                "n_folds",
                "mean_test_brier_prod",
                "mean_test_auc_prod",
                "mean_test_ap_prod",
                "chosen_stream_mode",
            ]
        ].copy()

        bt["mean_test_brier_prod"] = pd.to_numeric(bt["mean_test_brier_prod"], errors="coerce").map(_format_num)
        bt["mean_test_auc_prod"] = pd.to_numeric(bt["mean_test_auc_prod"], errors="coerce").map(
            lambda x: f"{float(x):.3f}" if x != "n/a" else x
        )
        bt["mean_test_ap_prod"] = pd.to_numeric(bt["mean_test_ap_prod"], errors="coerce").map(
            lambda x: f"{float(x):.3f}" if x != "n/a" else x
        )
        bt = bt.rename(
            columns={
                "horizon_days": "Horizon (days)",
                "n_folds": "Folds",
                "mean_test_brier_prod": "Mean Test Brier",
                "mean_test_auc_prod": "Mean Test AUC",
                "mean_test_ap_prod": "Mean Test AP",
                "chosen_stream_mode": "Chosen Stream",
            }
        )

        st.dataframe(bt, use_container_width=True, hide_index=True)

with lower_right:
    st.subheader("Model Gate Summary")
    if model_health.empty:
        st.info("No model health summary available yet.")
    else:
        mh = model_health.copy()
        mh["model_health_score"] = pd.to_numeric(mh["model_health_score"], errors="coerce")
        mh["strategy_edge_score"] = pd.to_numeric(mh["strategy_edge_score"], errors="coerce")
        mh = mh.sort_values(
            by=["model_gate", "strategy_edge_score", "model_health_score"],
            ascending=[True, False, False],
        )
        mh["model_health_score"] = mh["model_health_score"].map(_format_num)
        mh["strategy_edge_score"] = mh["strategy_edge_score"].map(_format_num)
        for col in [
            "evaluation_brier",
            "evaluation_auc",
            "evaluation_ap",
            "mean_test_brier_prod",
            "mean_test_auc_prod",
            "mean_test_ap_prod",
            "xgb_auc",
            "xgb_ap",
        ]:
            mh[col] = pd.to_numeric(mh[col], errors="coerce").map(
                lambda x: f"{float(x):.3f}" if pd.notna(x) else "n/a"
            )
        mh = mh.rename(
            columns={
                "event_rule": "Rule",
                "horizon_days": "Horizon (days)",
                "model_gate": "Gate",
                "decision_use": "Use",
                "model_health_score": "Health Score",
                "model_health_label": "Health",
                "strategy_edge_score": "Edge Score",
                "challenger_winner": "Challenger Winner",
                "evaluation_brier": "Eval Brier",
                "evaluation_auc": "Eval AUC",
                "evaluation_ap": "Eval AP",
                "mean_test_brier_prod": "Backtest Brier",
                "mean_test_auc_prod": "Backtest AUC",
                "mean_test_ap_prod": "Backtest AP",
                "chosen_stream_mode": "Chosen Stream",
                "xgb_auc": "XGB AUC",
                "xgb_ap": "XGB AP",
            }
        )
        st.dataframe(
            mh[
                [
                    "Rule",
                    "Horizon (days)",
                    "Gate",
                    "Use",
                    "Health",
                    "Health Score",
                    "Edge Score",
                    "Challenger Winner",
                    "Eval AUC",
                    "XGB AUC",
                    "Backtest AUC",
                    "Chosen Stream",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

# python main.py --skip-ui
# streamlit run ui/app.py
