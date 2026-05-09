from __future__ import annotations

import json
import os
from hashlib import md5
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st


GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"
GITHUB_API_VERSION = "2026-03-10"
DEFAULT_MODEL = os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4.1")

PAGE_TITLES = {
    "overview": "Overview",
    "forecast": "Forecast Console",
    "risk": "Risk Monitor",
    "calibration": "Calibration & Model Health",
    "regimes": "Regime View",
}

PAGE_SECTION_GUIDES = {
    "overview": [
        (
            "Signal Decision Board",
            "This is the shortlist of the most important signals. Each row is one model view of market downside risk, and the table helps the audience see which signals are stronger, weaker, or lower confidence.",
        ),
        (
            "Risk Signal History",
            "This chart shows how downside-risk estimates changed over time. Rising lines mean the system is seeing more market stress, while falling lines mean risk is easing.",
        ),
        (
            "SPY Price Overlay",
            "This compares price with one selected risk signal so you can see whether the risk estimate tends to climb during weaker market periods and soften during stronger periods.",
        ),
        (
            "Walk-Forward Backtest Snapshot",
            "This section summarizes how the models held up in historical out-of-sample tests. Lower Brier scores and higher AUC values are generally better.",
        ),
        (
            "Model Gate Summary",
            "This is the quality-control table. It separates models that are safer to discuss from models that should be treated as research-only or suppressed.",
        ),
    ],
    "forecast": [
        (
            "Forecast Ladder",
            "This ranks current signals from most decision-worthy to least. It is useful for seeing which horizon and event rule currently matter most.",
        ),
        (
            "Model Comparison",
            "This compares the interpretable logistic model with the XGBoost challenger so you can see where the nonlinear model adds value and where it does not.",
        ),
    ],
    "risk": [
        (
            "Downside Probability by Horizon",
            "This chart answers a simple question: is the system seeing more or less downside risk across short, medium, and longer horizons?",
        ),
        (
            "Shared-Window Price Overlay",
            "This overlay lines up price and the selected risk signal over the same time window so the comparison is fair and visually interpretable.",
        ),
        (
            "Latest Risk Snapshot",
            "This is the current state of each risk signal in one place. It is useful for quickly seeing posture, trend, and gate level.",
        ),
    ],
    "calibration": [
        (
            "Metric Chart",
            "This chart visualizes one reliability or performance metric across signals, making it easier to spot where a model is genuinely strong or only looks interesting at first glance.",
        ),
        (
            "Calibration Table",
            "This table explains whether the probabilities are trustworthy, not just whether the model is directional. It includes both logistic and XGBoost challenger metrics.",
        ),
    ],
    "regimes": [
        (
            "Regime Map",
            "This places signals in a market-state frame. It helps the audience interpret whether the market currently looks calm, cautious, or stressed.",
        ),
        (
            "Recent Market Backdrop",
            "This is the recent price context behind the current regime call, so non-technical viewers can connect the risk classification to visible market behavior.",
        ),
    ],
}

PAGE_CHART_GUIDES = {
    "overview": [
        "What does the risk signal history chart actually show?",
        "How should I read the SPY price overlay?",
        "What does it mean if the overlay risk line is rising while price is also rising?",
    ],
    "forecast": [
        "How should I read the forecast ladder?",
        "What does challenger winner mean in the comparison table?",
    ],
    "risk": [
        "What should I focus on in the downside probability chart?",
        "Why is the overlay chart drawn as a step line?",
        "What do the signal update markers mean?",
    ],
    "calibration": [
        "What does the metric ranking chart tell me?",
        "Why do AUC and Brier both matter?",
    ],
    "regimes": [
        "How should I read the regime map?",
        "What does a high z-score with high probability mean?",
    ],
}

GLOSSARY = {
    "Probability": "The estimated chance of a downside event over the stated horizon.",
    "Model Gate": "A quality-control label that says whether a signal is strong enough to trust, watch, research, or suppress.",
    "Health Score": "A composite score that rolls up reliability, discrimination, and calibration behavior into one easier number.",
    "AUC": "A ranking metric. Higher means the model is better at separating higher-risk cases from lower-risk ones.",
    "Brier": "A probability accuracy metric. Lower is better because it means the forecast probabilities were closer to reality.",
    "Backtest": "A historical simulation showing how a model would have behaved on unseen past data.",
    "Freshness": "How current the latest underlying market data is.",
    "Challenger Winner": "Whether the baseline logistic model or the XGBoost challenger looked better on the chosen metrics.",
}


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt_pct(value: Any, digits: int = 1) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}%}"


def _fmt_num(value: Any, digits: int = 1) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _github_token() -> str | None:
    return os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")


def _github_models_enabled() -> bool:
    return bool(_github_token())


def _top_signal_rows(signals: pd.DataFrame, top_n: int = 4) -> list[dict[str, Any]]:
    if signals.empty:
        return []

    frame = signals.copy()
    for col in ["event_probability", "model_health_score", "strategy_edge_score", "horizon_days"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "model_gate" not in frame.columns:
        frame["model_gate"] = "Unknown"
    frame["gate_rank"] = frame["model_gate"].map(
        {"Approved": 0, "Watch": 1, "Research Only": 2, "Suppressed": 3}
    ).fillna(4)
    frame = frame.sort_values(
        by=["gate_rank", "strategy_edge_score", "model_health_score", "event_probability"],
        ascending=[True, False, False, False],
    )
    keep_cols = [
        "event_rule",
        "horizon_days",
        "event_probability",
        "recommendation",
        "model_gate",
        "model_health_score",
        "challenger_winner",
        "freshness_status",
    ]
    return frame[keep_cols].head(top_n).to_dict(orient="records")


def _build_context(
    *,
    page_key: str,
    state: dict,
    signals: pd.DataFrame,
    history: pd.DataFrame,
    model_health: pd.DataFrame,
    backtests: pd.DataFrame,
) -> dict[str, Any]:
    summary = state.get("summary", {})
    consensus = state.get("consensus", {})

    trend_text = "n/a"
    if not history.empty:
        hist = history.copy()
        hist["as_of_date"] = pd.to_datetime(hist["as_of_date"], errors="coerce")
        hist["event_probability"] = pd.to_numeric(hist["event_probability"], errors="coerce")
        recent = hist.sort_values("as_of_date").dropna(subset=["as_of_date", "event_probability"]).tail(20)
        if len(recent) >= 2:
            delta = float(recent["event_probability"].iloc[-1] - recent["event_probability"].iloc[0])
            if delta >= 0.03:
                trend_text = "meaningfully rising"
            elif delta <= -0.03:
                trend_text = "meaningfully falling"
            else:
                trend_text = "fairly stable"

    return {
        "page_key": page_key,
        "page_title": PAGE_TITLES.get(page_key, page_key.title()),
        "summary": {
            "latest_as_of_date": summary.get("latest_as_of_date"),
            "freshness_status": summary.get("freshness_status"),
            "freshness_age_days": summary.get("freshness_age_days"),
            "consensus_action": summary.get("consensus_action"),
            "consensus_probability": summary.get("consensus_probability"),
            "average_health_score": summary.get("average_health_score"),
            "approved_signal_count": summary.get("approved_signal_count"),
            "suppressed_signal_count": summary.get("suppressed_signal_count"),
            "best_backtest_auc": summary.get("best_backtest_auc"),
            "best_xgb_auc": summary.get("best_xgb_auc"),
            "signal_agreement": summary.get("signal_agreement"),
            "xgb_win_count": summary.get("xgb_win_count"),
        },
        "consensus": {
            "signal_spread": consensus.get("signal_spread"),
        },
        "top_signals": _top_signal_rows(signals),
        "history_rows": int(len(history)),
        "history_trend": trend_text,
        "model_health_rows": int(len(model_health)),
        "backtest_rows": int(len(backtests)),
        "section_titles": [title for title, _ in PAGE_SECTION_GUIDES.get(page_key, [])],
    }


def _fallback_overview(context: dict[str, Any]) -> str:
    summary = context["summary"]
    top_signals = context["top_signals"]
    strongest = top_signals[0] if top_signals else None
    freshness = summary.get("freshness_status", "Unknown")
    latest_date = summary.get("latest_as_of_date", "n/a")
    consensus_action = summary.get("consensus_action", "n/a")
    consensus_prob = _fmt_pct(summary.get("consensus_probability"))
    agreement = summary.get("signal_agreement", "n/a")
    approved = int(summary.get("approved_signal_count", 0) or 0)
    suppressed = int(summary.get("suppressed_signal_count", 0) or 0)
    trend = context.get("history_trend", "n/a")

    freshness_line = (
        f"The latest market date reflected on this page is **{latest_date}**. "
        f"Data freshness is currently **{freshness}**"
    )
    age_days = summary.get("freshness_age_days")
    if age_days is not None and not pd.isna(age_days):
        freshness_line += f", which means the dashboard is working off information that is roughly **{int(age_days)} days** old."
    else:
        freshness_line += "."

    if strongest:
        strongest_text = (
            f"The most decision-relevant signal at the moment is **{strongest.get('event_rule', 'n/a')} | "
            f"{int(float(strongest.get('horizon_days', 0) or 0))}d**, with an estimated downside probability of "
            f"**{_fmt_pct(strongest.get('event_probability'))}**. It is currently classified as **{strongest.get('model_gate', 'n/a')}**, "
            f"and the suggested posture is **{strongest.get('recommendation', 'n/a')}**."
        )
    else:
        strongest_text = (
            "There is no single dominant signal available on this page right now, so the current read should be treated as a blended monitoring view."
        )

    return "\n\n".join(
        [
            "### Executive Summary",
            (
                f"This page is the **{context['page_title']}** view of the Market Risk Engine. "
                f"At a high level, the system’s current message is **{consensus_action}**, based on a consensus downside probability of **{consensus_prob}**."
            ),
            "### What Matters Right Now",
            strongest_text,
            (
                f"Across the current signal set, agreement is **{agreement}** and the recent risk trend has been **{trend}**. "
                f"There are **{approved} approved** signals and **{suppressed} suppressed** signals, which helps separate the models worth emphasizing from those that should stay in the background."
            ),
            "### Data Context",
            freshness_line,
            (
                "This summary is designed to translate model output into plain-language guidance for decision support. "
                "It should be read as a structured market-risk briefing, not as guaranteed investment advice."
            ),
        ]
    )


def _fallback_answer(question: str, context: dict[str, Any]) -> str:
    q = question.lower()
    summary = context["summary"]
    top_signals = context["top_signals"]

    if "fresh" in q or "stale" in q or "update" in q:
        return (
            f"The dashboard freshness status is **{summary.get('freshness_status', 'Unknown')}**. "
            f"The latest market date is **{summary.get('latest_as_of_date', 'n/a')}**, which is about "
            f"**{summary.get('freshness_age_days', 'n/a')}** days old. Freshness matters because even a good model is less useful if the underlying market data is old."
        )
    if "gate" in q or "approved" in q or "suppressed" in q:
        return (
            f"Model gates are quality-control labels. Right now there are **{summary.get('approved_signal_count', 0)} approved** signals "
            f"and **{summary.get('suppressed_signal_count', 0)} suppressed** signals. "
            "Approved means a signal cleared more of the reliability checks, while suppressed means it should not be emphasized for decision-making."
        )
    if "xgboost" in q or "challenger" in q:
        return (
            f"XGBoost is the nonlinear challenger model. In the current dashboard it records **{summary.get('xgb_win_count', 0)} wins** across the compared signals. "
            "It is there to test whether a more flexible model can add value beyond the baseline logistic model."
        )
    if "probab" in q or "what does this mean" in q or "risk" in q:
        if top_signals:
            strongest = top_signals[0]
            return (
                "A risk probability is the model’s estimate of how likely a downside event is over a stated horizon. "
                f"The strongest current signal is **{strongest.get('event_rule', 'n/a')} | {int(float(strongest.get('horizon_days', 0) or 0))}d** "
                f"at **{_fmt_pct(strongest.get('event_probability'))}**. That does **not** mean a decline is guaranteed; it means the model currently sees elevated downside odds relative to normal conditions."
            )
    if "auc" in q or "brier" in q or "calibration" in q:
        return (
            "The technical metrics are there to answer a simple question: can we trust the probabilities? "
            "**AUC** measures how well the model separates higher-risk from lower-risk cases. "
            "**Brier** measures how close the forecast probabilities were to what actually happened, so lower is better. "
            "**Calibration** checks whether the model’s probabilities are honest rather than overstated."
        )

    return (
        _fallback_overview(context)
        + "\n\nIf you want a more targeted explanation, try asking about freshness, model gates, XGBoost, calibration, or what a specific section means."
    )


def _call_github_models(messages: list[dict[str, str]]) -> str | None:
    token = _github_token()
    if not token:
        return None

    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 700,
    }
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(GITHUB_MODELS_URL, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        data = response.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
    except Exception:
        return None


def _ai_overview(context: dict[str, Any]) -> str:
    system = (
        "You explain market-risk dashboards to non-technical audiences. "
        "Use plain English, short paragraphs, and avoid jargon unless you define it. "
        "Do not give personal investment advice. Mention data freshness if it is not fresh."
    )
    user = (
        "Summarize this dashboard page for a smart but non-technical audience. "
        "Explain what the page is for, what matters most right now, and how to read it. "
        "Keep it concise but helpful.\n\n"
        f"Context:\n{json.dumps(context, default=str)}"
    )
    ai_text = _call_github_models(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return ai_text or _fallback_overview(context)


def _ai_answer(question: str, context: dict[str, Any]) -> str:
    system = (
        "You are a plain-English dashboard copilot for a market-risk project. "
        "Explain only from the provided context. Be clear, grounded, and non-technical. "
        "Do not give guaranteed investment advice."
    )
    user = (
        f"Question: {question}\n\n"
        "Answer for a non-technical audience using only this dashboard context:\n"
        f"{json.dumps(context, default=str)}"
    )
    ai_text = _call_github_models(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return ai_text or _fallback_answer(question, context)


def _context_key(page_key: str, context: dict[str, Any]) -> str:
    digest = md5(json.dumps(context, default=str, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{page_key}_{digest}"


def render_ai_explainer(
    *,
    page_key: str,
    state: dict,
    signals: pd.DataFrame,
    history: pd.DataFrame,
    model_health: pd.DataFrame,
    backtests: pd.DataFrame,
) -> None:
    context = _build_context(
        page_key=page_key,
        state=state,
        signals=signals,
        history=history,
        model_health=model_health,
        backtests=backtests,
    )
    cache_key = _context_key(page_key, context)
    overview_state_key = f"copilot_overview_{cache_key}"
    chat_state_key = f"copilot_chat_{page_key}"

    if overview_state_key not in st.session_state:
        st.session_state[overview_state_key] = _ai_overview(context)

    if chat_state_key not in st.session_state:
        st.session_state[chat_state_key] = [
            {
                "role": "assistant",
                "content": (
                    f"I’m the **Copilot Explainer** for the {PAGE_TITLES.get(page_key, page_key)} page. "
                    "Ask me what a section means, whether the data is fresh, or how to interpret the current signals."
                ),
            }
        ]

    st.divider()
    st.subheader("Copilot Explainer")
    if _github_models_enabled():
        st.caption(
            f"Live GitHub Models mode is enabled using `{DEFAULT_MODEL}`. "
            "If the API is unavailable, the dashboard will automatically fall back to the built-in explainer."
        )
    else:
        st.caption(
            "Built-in explainer mode is active for this session. "
            "You can optionally add `GITHUB_TOKEN` or `GH_TOKEN` later to enable live GitHub Models summaries and chat responses."
        )

    tab_summary, tab_sections, tab_chat = st.tabs(
        ["Page Summary", "Section Guide", "Ask the Dashboard"]
    )

    with tab_summary:
        st.markdown(st.session_state[overview_state_key])

    with tab_sections:
        for title, base_text in PAGE_SECTION_GUIDES.get(page_key, []):
            with st.expander(title, expanded=False):
                st.markdown(base_text)
        with st.expander("Glossary", expanded=False):
            for term, meaning in GLOSSARY.items():
                st.markdown(f"**{term}**: {meaning}")

    with tab_chat:
        for message in st.session_state[chat_state_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        st.caption(
            "This chart copilot can explain what each visualization means in plain English and help interpret the current readings."
        )
        suggestions = PAGE_CHART_GUIDES.get(page_key, [])
        if suggestions:
            cols = st.columns(min(2, len(suggestions)))
            for i, suggestion in enumerate(suggestions[:4]):
                with cols[i % len(cols)]:
                    if st.button(suggestion, key=f"{page_key}_chart_prompt_{i}", use_container_width=True):
                        st.session_state[f"queued_prompt_{page_key}"] = suggestion

        queued_prompt_key = f"queued_prompt_{page_key}"
        prompt = st.chat_input(
            f"Ask the Copilot explainer about the charts on the {PAGE_TITLES.get(page_key, page_key).lower()} page...",
            key=f"copilot_input_{page_key}",
        )
        if not prompt and queued_prompt_key in st.session_state:
            prompt = st.session_state.pop(queued_prompt_key)
        if prompt:
            st.session_state[chat_state_key].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            answer = _ai_answer(prompt, context)
            st.session_state[chat_state_key].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
