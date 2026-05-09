from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def apply_clean_chart_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0.96)",
        plot_bgcolor="rgba(255,255,255,0.96)",
        font=dict(color="#24324b"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(36, 50, 75, 0.10)",
        zeroline=False,
    )
    return fig


def build_downside_history_chart(
    history: pd.DataFrame,
    *,
    title: str,
    smoothing_window: int = 21,
    recent_years: int = 3,
) -> go.Figure:
    hist = history.copy()
    hist["as_of_date"] = pd.to_datetime(hist["as_of_date"], errors="coerce")
    hist["event_probability"] = pd.to_numeric(hist["event_probability"], errors="coerce")
    hist["horizon_days"] = pd.to_numeric(hist["horizon_days"], errors="coerce")
    hist = hist.dropna(subset=["as_of_date", "event_probability", "horizon_days"]).copy()
    hist["horizon_days"] = hist["horizon_days"].astype(int)

    if recent_years > 0 and not hist.empty:
        end_date = hist["as_of_date"].max()
        start_date = end_date - pd.Timedelta(days=int(recent_years * 365))
        hist = hist[hist["as_of_date"] >= start_date].copy()

    hist = hist.sort_values(["event_rule", "horizon_days", "as_of_date"])
    hist["smoothed_probability"] = (
        hist.groupby(["event_rule", "horizon_days"])["event_probability"]
        .transform(lambda s: s.rolling(max(1, int(smoothing_window)), min_periods=1).mean())
    )
    hist["rule_label"] = hist["event_rule"].astype(str).str.title()
    hist["horizon_label"] = hist["horizon_days"].astype(str) + "d Horizon"

    fig = px.line(
        hist,
        x="as_of_date",
        y="smoothed_probability",
        color="rule_label",
        facet_row="horizon_label",
        title=title,
        labels={
            "as_of_date": "Date",
            "smoothed_probability": "Smoothed Probability",
            "rule_label": "Rule",
            "horizon_label": "Horizon",
        },
        color_discrete_map={
            "Quantile": "#2979c9",
            "Sigma": "#c06a28",
        },
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(line=dict(width=2.5))
    return apply_clean_chart_style(fig)


def build_overlay_chart(
    shared_prices: pd.DataFrame,
    overlay_daily: pd.DataFrame,
    update_points: pd.DataFrame,
    *,
    risk_name: str,
    show_markers: bool,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=shared_prices["date"],
            y=shared_prices["close"],
            name="SPY Close",
            line=dict(color="#1f3b5d", width=2.4),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=overlay_daily["date"],
            y=overlay_daily["event_probability"],
            name=risk_name,
            line=dict(color="#c06a28", width=3),
            line_shape="hv",
            yaxis="y2",
        )
    )
    if show_markers:
        fig.add_trace(
            go.Scatter(
                x=update_points["as_of_date"],
                y=update_points["event_probability"],
                name="Signal Updates",
                mode="markers",
                marker=dict(color="#c06a28", size=6, opacity=0.75),
                yaxis="y2",
            )
        )
    fig.update_layout(
        yaxis=dict(title="SPY Close"),
        yaxis2=dict(title="Risk Probability", overlaying="y", side="right", tickformat=".0%"),
    )
    return apply_clean_chart_style(fig)


def build_metric_rank_chart(
    plot_df: pd.DataFrame,
    *,
    metric: str,
    title: str,
) -> go.Figure:
    frame = plot_df.copy().sort_values(metric, ascending=False)
    fig = px.bar(
        frame,
        x=metric,
        y="label",
        color="model_gate",
        orientation="h",
        title=title,
        labels={metric: metric.replace("_", " ").title(), "label": "Signal"},
        color_discrete_map={
            "Approved": "#2e8b57",
            "Watch": "#d8a12d",
            "Research Only": "#c06a28",
            "Suppressed": "#b13d3d",
        },
    )
    return apply_clean_chart_style(fig)


def build_regime_scatter(regime_df: pd.DataFrame) -> go.Figure:
    frame = regime_df.copy()
    if "model_health_score" in frame.columns:
        frame["marker_size"] = pd.to_numeric(frame["model_health_score"], errors="coerce").fillna(40)
    else:
        frame["marker_size"] = 40

    fig = px.scatter(
        frame,
        x="signal_zscore",
        y="event_probability",
        color="model_gate",
        size="marker_size",
        hover_name="label",
        title="Signals by Z-Score and Probability",
        labels={"signal_zscore": "Signal Z-Score", "event_probability": "Probability"},
        color_discrete_map={
            "Approved": "#2e8b57",
            "Watch": "#d8a12d",
            "Research Only": "#c06a28",
            "Suppressed": "#b13d3d",
        },
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0)))
    fig.update_yaxes(tickformat=".0%")
    fig.add_hline(y=0.16, line_dash="dot", line_color="rgba(36, 50, 75, 0.35)")
    fig.add_hline(y=0.24, line_dash="dot", line_color="rgba(36, 50, 75, 0.35)")
    fig.add_vline(x=0.0, line_dash="dot", line_color="rgba(36, 50, 75, 0.35)")
    return apply_clean_chart_style(fig)
