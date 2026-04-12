from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.reporting.product_summary import build_dashboard_state, write_product_outputs


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def outputs_dir() -> Path:
    out = repo_root() / "artifacts" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_dashboard_state(market: str = "SPY") -> dict:
    state_path = outputs_dir() / f"dashboard_state_{market}_latest.json"

    if not state_path.exists():
        write_product_outputs(market=market, as_of_date="latest", repo_root=repo_root())

    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return build_dashboard_state(market=market, repo_root=repo_root())


def load_signal_frames(
    market: str = "SPY",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state = load_dashboard_state(market=market)

    signals = pd.DataFrame(state.get("signals", []))
    history = pd.DataFrame(state.get("signal_history", []))
    model_health = pd.DataFrame(state.get("model_health", []))
    backtests = pd.DataFrame(state.get("backtests", []))

    for df in [signals, history]:
        if not df.empty and "as_of_date" in df.columns:
            df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")

    return signals, history, model_health, backtests


def load_spy_prices() -> pd.DataFrame:
    price_path = outputs_dir() / "spy_prices.csv"
    if not price_path.exists():
        return pd.DataFrame(columns=["date", "close"])

    df = pd.read_csv(price_path)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def load_artifact_csv(artifact_dir: str, filename: str) -> pd.DataFrame:
    if not artifact_dir:
        return pd.DataFrame()

    path = Path(artifact_dir) / filename
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_artifact_json(artifact_dir: str, filename: str) -> dict:
    if not artifact_dir:
        return {}

    path = Path(artifact_dir) / filename
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
