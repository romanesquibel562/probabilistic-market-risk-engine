from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.ai_explainer import render_ai_explainer  # noqa: E402
from components.chart_utils import build_downside_history_chart, build_overlay_chart  # noqa: E402
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


def _overlay_signal_label(event_rule: str, horizon_days: int, gate: str | None = None) -> str:
    suffix = f" [{gate}]" if gate else ""
    return f"{str(event_rule)} | {int(horizon_days)}d{suffix}"


st.sidebar.title("Market Risk Engine")
st.sidebar.caption("Decision-support dashboard for downside-risk monitoring")

market = st.sidebar.selectbox("Market", ["SPY"], index=0)
show_research_only = st.sidebar.toggle("Show research-only signals", value=True)
selected_horizons = st.sidebar.multiselect(
    "Horizons",
    [5, 21, 63],
    default=[5, 21, 63],
)
history_window_years = st.sidebar.selectbox("History window", [1, 2, 3, 5], index=2)
history_smoothing = st.sidebar.slider("History smoothing (days)", min_value=5, max_value=63, value=21, step=4)

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

overlay_candidates = signals.copy()
overlay_candidates["overlay_priority"] = overlay_candidates["model_gate"].map(
    {"Approved": 0, "Watch": 1, "Research Only": 2, "Suppressed": 3}
).fillna(4)
overlay_candidates = overlay_candidates.sort_values(
    by=["overlay_priority", "strategy_edge_score", "model_health_score", "event_probability"],
    ascending=[True, False, False, False],
)
overlay_options = []
overlay_lookup: dict[str, tuple[str, int]] = {}
for row in overlay_candidates.itertuples():
    if pd.isna(row.horizon_days):
        continue
    label = _overlay_signal_label(row.event_rule, int(row.horizon_days), row.model_gate)
    if label not in overlay_lookup:
        overlay_lookup[label] = (str(row.event_rule), int(row.horizon_days))
        overlay_options.append(label)

default_overlay = next(
    (label for label in overlay_options if label.startswith("sigma | 21d")),
    overlay_options[0] if overlay_options else None,
)
selected_overlay_label = st.sidebar.selectbox(
    "Overlay signal",
    overlay_options,
    index=overlay_options.index(default_overlay) if default_overlay in overlay_options else 0,
) if overlay_options else None
selected_overlay = overlay_lookup.get(selected_overlay_label, ("sigma", 21))
overlay_window = st.sidebar.selectbox("Overlay window", ["1Y", "2Y", "3Y", "Full"], index=1)
show_overlay_markers = st.sidebar.toggle("Show overlay markers", value=False)

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
        st.caption(
            "Smoothed small-multiple view. Each row is one horizon, and the lines show the underlying risk rules without the clutter of every raw point."
        )
        fig = build_downside_history_chart(
            history,
            title="Multi-Horizon Downside Probability",
            smoothing_window=history_smoothing,
            recent_years=history_window_years,
        )
        st.plotly_chart(fig, use_container_width=True)

with chart_right:
    st.subheader("SPY Price Overlay")
    if prices.empty or history.empty:
        st.info("Price overlay unavailable.")
    else:
        overlay = history.copy()
        overlay = overlay.sort_values("as_of_date")
        overlay_rule, overlay_horizon = selected_overlay
        overlay_series = overlay[
            (overlay["event_rule"].astype(str) == overlay_rule)
            & (pd.to_numeric(overlay["horizon_days"], errors="coerce") == int(overlay_horizon))
        ][["as_of_date", "event_probability"]].copy()
        overlay_series["event_probability"] = pd.to_numeric(
            overlay_series["event_probability"], errors="coerce"
        )
        overlay_series = overlay_series.dropna(subset=["as_of_date", "event_probability"]).copy()

        overlay_meta = signals[
            (signals["event_rule"].astype(str) == overlay_rule)
            & (pd.to_numeric(signals["horizon_days"], errors="coerce") == int(overlay_horizon))
        ].head(1)

        if overlay_series.empty:
            st.info("No overlay risk series is available yet for the selected signal.")
        else:
            shared_start = overlay_series["as_of_date"].min()
            shared_end = overlay_series["as_of_date"].max()
            if overlay_window != "Full":
                years = {"1Y": 365, "2Y": 730, "3Y": 1095}[overlay_window]
                shared_start = max(shared_start, shared_end - pd.Timedelta(days=years))
                overlay_series = overlay_series[overlay_series["as_of_date"] >= shared_start].copy()
            shared_prices = prices[
                (prices["date"] >= shared_start) & (prices["date"] <= shared_end)
            ].copy()

            if shared_prices.empty:
                st.info("No overlapping SPY price window is available for the current risk series.")
            else:
                overlay_daily = pd.merge_asof(
                    shared_prices[["date"]].sort_values("date"),
                    overlay_series.rename(columns={"as_of_date": "date"}).sort_values("date"),
                    on="date",
                    direction="backward",
                ).dropna(subset=["event_probability"])

                update_points = overlay_series.copy()
                update_points["delta"] = update_points["event_probability"].diff().abs()
                update_points = update_points[
                    update_points["delta"].fillna(1.0) >= 0.01
                ].drop(columns=["delta"])
                if update_points.empty:
                    update_points = overlay_series.tail(min(12, len(overlay_series))).copy()

                if not overlay_meta.empty:
                    meta = overlay_meta.iloc[0]
                    st.caption(
                        f"Shared-window overlay from {shared_start.strftime('%Y-%m-%d')} "
                        f"to {shared_end.strftime('%Y-%m-%d')}. "
                        f"Current signal: {_format_pct(meta.get('event_probability'))} | "
                        f"Gate: {meta.get('model_gate', 'n/a')} | "
                        f"Health: {_format_num(meta.get('model_health_score'))}. "
                        "Risk is held at the last known model estimate until the next update."
                    )
                else:
                    st.caption(
                        f"Shared-window overlay from {shared_start.strftime('%Y-%m-%d')} "
                        f"to {shared_end.strftime('%Y-%m-%d')}. Risk is held at the last known model estimate "
                        "until the next update."
                    )

                price_fig = build_overlay_chart(
                    shared_prices,
                    overlay_daily,
                    update_points,
                    risk_name=f"{overlay_horizon}d {str(overlay_rule).title()} Risk",
                    show_markers=show_overlay_markers,
                )
                price_fig.update_xaxes(range=[shared_start, shared_end])
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

render_ai_explainer(
    page_key="overview",
    state=state,
    signals=signals,
    history=history,
    model_health=model_health,
    backtests=backtests,
)

# python main.py --skip-ui
# streamlit run ui/app.py
