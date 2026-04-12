from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.data_loader import load_signal_frames  # noqa: E402


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
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(81, 57, 38, 0.12);
        padding: 0.9rem 1rem;
        border-radius: 16px;
    }

    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Market Risk Engine")
st.sidebar.caption("Interactive market risk intelligence dashboard")

market = st.sidebar.selectbox("Market", ["SPY"], index=0)

signals, history, model_health, backtests = load_signal_frames(market=market)

st.title("Market Risk Intelligence Dashboard")
st.caption(
    "Decision-oriented overview of latest risk signals, model health, and walk-forward usefulness."
)

if signals.empty:
    st.warning("No dashboard signals found yet. Run `python main.py --skip-ui` first.")
    st.stop()

latest_date = signals["as_of_date"].max()
avg_health = float(pd.to_numeric(signals["model_health_score"], errors="coerce").mean())
avg_prob = float(pd.to_numeric(signals["event_probability"], errors="coerce").mean())
elevated = int(signals["recommendation"].isin(["High Alert", "Defensive"]).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Latest Signal Date",
    latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "n/a",
)
c2.metric("Elevated Signals", elevated)
c3.metric("Average Health Score", f"{avg_health:.1f}")
c4.metric("Average Probability", f"{avg_prob:.1%}")

st.subheader("Latest Signal Board")

board = signals[
    [
        "event_rule",
        "horizon_days",
        "event_probability",
        "probability_delta_1obs",
        "historical_percentile",
        "recommendation",
        "model_health_label",
        "model_health_score",
    ]
].copy()

board["event_probability"] = pd.to_numeric(
    board["event_probability"], errors="coerce"
).map(lambda x: f"{x:.1%}" if pd.notna(x) else "n/a")

board["probability_delta_1obs"] = pd.to_numeric(
    board["probability_delta_1obs"], errors="coerce"
).map(lambda x: f"{x:+.1%}" if pd.notna(x) else "n/a")

board["historical_percentile"] = pd.to_numeric(
    board["historical_percentile"], errors="coerce"
).map(lambda x: f"{x:.0%}" if pd.notna(x) else "n/a")

board["model_health_score"] = pd.to_numeric(
    board["model_health_score"], errors="coerce"
).map(lambda x: f"{x:.1f}" if pd.notna(x) else "n/a")

board = board.rename(
    columns={
        "event_rule": "Rule",
        "horizon_days": "Horizon (days)",
        "event_probability": "Current Probability",
        "probability_delta_1obs": "1-Step Change",
        "historical_percentile": "Historical Percentile",
        "recommendation": "Decision Posture",
        "model_health_label": "Health",
        "model_health_score": "Health Score",
    }
)

st.dataframe(board, use_container_width=True, hide_index=True)

if not history.empty:
    st.subheader("Risk Probability History")

    hist = history.copy()
    hist["series"] = (
        hist["event_rule"].astype(str)
        + " | "
        + hist["horizon_days"].astype(str)
        + "d"
    )

    fig = px.line(
        hist,
        x="as_of_date",
        y="event_probability",
        color="series",
        markers=True,
        title="Probability Trend Across Horizons and Event Families",
        labels={
            "as_of_date": "Date",
            "event_probability": "Probability",
            "series": "Signal",
        },
    )
    fig.update_layout(
        legend_title_text="Signal",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1, 1])

with left:
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

with right:
    st.subheader("Model Health Summary")
    if model_health.empty:
        st.info("No model health summary available yet.")
    else:
        mh = model_health.copy()
        mh = mh.rename(
            columns={
                "event_rule": "Rule",
                "horizon_days": "Horizon (days)",
                "model_health_score": "Health Score",
                "model_health_label": "Health",
                "evaluation_brier": "Eval Brier",
                "evaluation_auc": "Eval AUC",
                "evaluation_ap": "Eval AP",
                "evaluation_drift_gap_abs_mean": "Mean |Drift Gap|",
                "evaluation_topk_precision": "Top-K Precision",
                "evaluation_topk_recall": "Top-K Recall",
                "evaluation_topk_f1": "Top-K F1",
                "mean_test_brier_prod": "Backtest Brier",
                "mean_test_auc_prod": "Backtest AUC",
                "mean_test_ap_prod": "Backtest AP",
                "chosen_stream_mode": "Chosen Stream",
            }
        )
        st.dataframe(mh, use_container_width=True, hide_index=True)




# streamlit run ui/app.py

