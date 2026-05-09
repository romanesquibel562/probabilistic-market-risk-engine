from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.ai_explainer import render_ai_explainer  # noqa: E402
from components.chart_utils import apply_clean_chart_style, build_regime_scatter  # noqa: E402
from components.data_loader import load_signal_frames, load_spy_prices  # noqa: E402


def _regime_label(prob: float, spread: float) -> str:
    if prob >= 0.24:
        return "Stress"
    if prob >= 0.16 or spread >= 0.10:
        return "Caution"
    return "Calm"


st.title("Regime View")
st.caption("A lightweight regime layer built from consensus downside risk, dispersion, and recent market movement.")

state, signals, history, model_health, backtests = load_signal_frames()
prices = load_spy_prices()
summary = state.get("summary", {})

if signals.empty:
    st.warning("No signal data found. Run `python main.py --skip-ui` first.")
    st.stop()

signals = signals.copy()
signals["event_probability"] = pd.to_numeric(signals["event_probability"], errors="coerce")
signals["horizon_days"] = pd.to_numeric(signals["horizon_days"], errors="coerce")
signals["signal_zscore"] = pd.to_numeric(signals.get("signal_zscore"), errors="coerce")

consensus_prob = float(summary.get("consensus_probability", np.nan))
signal_spread = float(state.get("consensus", {}).get("signal_spread", np.nan))
regime = _regime_label(consensus_prob, signal_spread)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Regime", regime)
c2.metric("Consensus Probability", f"{consensus_prob:.1%}" if pd.notna(consensus_prob) else "n/a")
c3.metric("Signal Spread", f"{signal_spread:.1%}" if pd.notna(signal_spread) else "n/a")
c4.metric("Agreement", summary.get("signal_agreement", "n/a"))

st.subheader("Regime Map")
regime_df = signals[
    ["event_rule", "horizon_days", "event_probability", "signal_zscore", "model_gate", "model_health_score"]
].copy()
regime_df["label"] = regime_df["event_rule"].astype(str) + " | " + regime_df["horizon_days"].astype(str) + "d"
st.caption("This version removes on-chart text clutter and uses color and size to show model quality more clearly.")
fig = build_regime_scatter(regime_df)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent Market Backdrop")
if prices.empty:
    st.info("No price history available.")
else:
    recent = prices.sort_values("date").copy()
    recent["ret_5d"] = recent["close"].pct_change(5)
    recent["ret_21d"] = recent["close"].pct_change(21)
    tail = recent.tail(60).copy()
    import plotly.graph_objects as go

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=tail["date"],
            y=tail["close"],
            name="SPY Close",
            line=dict(color="#1f3b5d", width=2.5),
        )
    )
    fig2.update_layout(title="Recent SPY Price Context", yaxis=dict(title="SPY Close"))
    fig2 = apply_clean_chart_style(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    latest = recent.iloc[-1]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "5d Return": f"{float(latest['ret_5d']):.1%}" if pd.notna(latest["ret_5d"]) else "n/a",
                    "21d Return": f"{float(latest['ret_21d']):.1%}" if pd.notna(latest["ret_21d"]) else "n/a",
                    "Latest Close": f"{float(latest['close']):.2f}",
                    "Interpretation": regime,
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

render_ai_explainer(
    page_key="regimes",
    state=state,
    signals=signals,
    history=history,
    model_health=model_health,
    backtests=backtests,
)
