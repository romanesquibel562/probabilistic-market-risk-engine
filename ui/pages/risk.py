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
from components.chart_utils import build_downside_history_chart, build_overlay_chart  # noqa: E402
from components.data_loader import load_signal_frames, load_spy_prices  # noqa: E402


st.title("Risk Monitor")
st.caption("Signal history and price-aligned risk overlays across the shared live window.")

state, signals, history, model_health, backtests = load_signal_frames()
prices = load_spy_prices()

if history.empty:
    st.warning("No history found. Run `python main.py --skip-ui` first.")
    st.stop()

history = history.copy()
history["event_probability"] = pd.to_numeric(history["event_probability"], errors="coerce")
history["horizon_days"] = pd.to_numeric(history["horizon_days"], errors="coerce")

horizon = st.selectbox("Overlay Horizon", [5, 21, 63], index=1)
rule = st.selectbox("Primary Risk Rule", sorted(history["event_rule"].dropna().unique().tolist()), index=0)
overlay_window = st.selectbox("Overlay Window", ["1Y", "2Y", "3Y", "Full"], index=1)
history_window_years = st.selectbox("History Window", [1, 2, 3, 5], index=2)
history_smoothing = st.slider("History Smoothing (days)", min_value=5, max_value=63, value=21, step=4)
show_markers = st.toggle("Show Signal Update Markers", value=False)
st.caption(
    "This chart uses a smoothed recent window and separates horizons into rows so the signal structure is easier to compare."
)
fig = build_downside_history_chart(
    history,
    title="Downside Probability by Horizon",
    smoothing_window=history_smoothing,
    recent_years=history_window_years,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Shared-Window Price Overlay")
subset = history[
    (history["event_rule"].astype(str) == str(rule))
    & (history["horizon_days"] == int(horizon))
][["as_of_date", "event_probability"]].copy()
subset = subset.dropna(subset=["as_of_date", "event_probability"]).sort_values("as_of_date")

if subset.empty or prices.empty:
    st.info("Not enough overlapping data for the selected overlay.")
else:
    shared_start = subset["as_of_date"].min()
    shared_end = subset["as_of_date"].max()
    if overlay_window != "Full":
        years = {"1Y": 365, "2Y": 730, "3Y": 1095}[overlay_window]
        shared_start = max(shared_start, shared_end - pd.Timedelta(days=years))
        subset = subset[subset["as_of_date"] >= shared_start].copy()
    shared_prices = prices[(prices["date"] >= shared_start) & (prices["date"] <= shared_end)].copy()

    if shared_prices.empty:
        st.info("No overlapping price window available.")
    else:
        st.caption(
            f"Comparing SPY close and {rule} {horizon}d risk over the shared window "
            f"{shared_start.strftime('%Y-%m-%d')} to {shared_end.strftime('%Y-%m-%d')}."
        )
        overlay_daily = pd.merge_asof(
            shared_prices[["date"]].sort_values("date"),
            subset.rename(columns={"as_of_date": "date"}).sort_values("date"),
            on="date",
            direction="backward",
        ).dropna(subset=["event_probability"])
        update_points = subset.copy()
        update_points["delta"] = update_points["event_probability"].diff().abs()
        update_points = update_points[update_points["delta"].fillna(1.0) >= 0.01].drop(columns=["delta"])
        if update_points.empty:
            update_points = subset.tail(min(12, len(subset))).copy()
        overlay = build_overlay_chart(
            shared_prices,
            overlay_daily,
            update_points,
            risk_name=f"{rule} {horizon}d Risk",
            show_markers=show_markers,
        )
        overlay.update_xaxes(range=[shared_start, shared_end])
        st.plotly_chart(overlay, use_container_width=True)

st.subheader("Latest Risk Snapshot")
latest = signals.copy()
latest["event_probability"] = pd.to_numeric(latest["event_probability"], errors="coerce")
latest["horizon_days"] = pd.to_numeric(latest["horizon_days"], errors="coerce")
latest = latest.sort_values(["horizon_days", "event_rule"])
latest["event_probability"] = latest["event_probability"].map(lambda x: f"{float(x):.1%}" if pd.notna(x) else "n/a")
latest = latest.rename(
    columns={
        "event_rule": "Rule",
        "horizon_days": "Horizon (days)",
        "event_probability": "Probability",
        "signal_trend": "Trend",
        "historical_percentile": "Percentile",
        "recommendation": "Posture",
    }
)
st.dataframe(
    latest[["Rule", "Horizon (days)", "Probability", "Trend", "Posture", "model_gate"]],
    use_container_width=True,
    hide_index=True,
)

render_ai_explainer(
    page_key="risk",
    state=state,
    signals=signals,
    history=history,
    model_health=model_health,
    backtests=backtests,
)
