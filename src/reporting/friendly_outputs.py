# src/reporting/friendly_outputs.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class FriendlySummaryConfig:
    market: str = "SPY"
    out_dir: str = "artifacts/outputs"
    artifact_root: str = "artifacts/models"

    # Which rolling windows to prefer (matches what your logs already show)
    preferred_windows_by_horizon: dict[int, int] | None = None

    # Simple, explainable alert rule (tweak later)
    elevated_horizons: tuple[int, ...] = (63,)
    elevated_prob_threshold: float = 0.25


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_latest_artifact_dir(
    *,
    artifact_root: Path,
    market: str,
    target_name: str,
    horizon_days: int,
) -> Path:
    """
    Finds the most recently modified artifact directory for a given (market, target, horizon).
    Expected directory naming (from your runs):
      artifacts/models/eventlogit_..._{MARKET}_{TARGET}_h{H}d_YYYYMMDD_HHMMSS
    """
    if not artifact_root.exists():
        raise FileNotFoundError(f"artifact_root not found: {artifact_root.as_posix()}")

    key = f"_{market}_{target_name}_h{int(horizon_days)}d_"
    candidates: list[Path] = []
    for p in artifact_root.iterdir():
        if p.is_dir() and key in p.name:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No artifact dirs found for market={market} target={target_name} h={horizon_days} under {artifact_root.as_posix()}.\n"
            f"Expected a folder name containing: {key}"
        )

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_rolling_drift(artifact_dir: Path) -> pd.DataFrame:
    """
    Your risk_event_logistic run prints a "Rolling Calibration Drift" table.
    This function expects you saved it as a CSV inside the artifact directory.
    We accept a few possible filenames to be robust.
    """
    candidates = [
        artifact_dir / "rolling_calibration_drift.csv",
        artifact_dir / "rolling_calibration.csv",
        artifact_dir / "calibration_drift.csv",
        artifact_dir / "rolling_drift.csv",
    ]
    for fp in candidates:
        if fp.exists():
            df = pd.read_csv(fp)
            # expected columns (based on what your logs print)
            # as_of_date,label,window,n_window,roll_event_rate,roll_p_mean,roll_brier,roll_logloss
            return df

    raise FileNotFoundError(
        "Could not find rolling calibration drift CSV in artifact dir:\n"
        f"  {artifact_dir.as_posix()}\n"
        "Looked for one of:\n"
        + "\n".join([f"  - {c.name}" for c in candidates])
        + "\n\n"
        "Fix: make sure risk_event_logistic writes the printed drift table to 'rolling_calibration_drift.csv'."
    )


def _pick_best_row_for_horizon(
    drift: pd.DataFrame,
    *,
    horizon_days: int,
    preferred_window: int,
) -> pd.Series:
    """
    Picks the row we use to publish:
      - label should start with 'test_prod'
      - choose preferred_window if available, otherwise choose the largest window available
    """
    if drift.empty:
        raise ValueError("rolling drift df is empty")

    d = drift.copy()
    d["as_of_date"] = pd.to_datetime(d["as_of_date"])
    d = d.sort_values("as_of_date").reset_index(drop=True)

    # keep test_prod rows
    d = d[d["label"].astype(str).str.startswith("test_prod")].copy()
    if d.empty:
        raise ValueError("rolling drift has no rows with label starting with 'test_prod'")

    # prefer specific window if present, else fallback to max window present
    if "window" not in d.columns:
        raise ValueError("rolling drift is missing required column: window")

    d["window"] = pd.to_numeric(d["window"], errors="coerce").astype("Int64")
    if (d["window"] == preferred_window).any():
        d = d[d["window"] == preferred_window].copy()
    else:
        maxw = int(d["window"].dropna().max())
        d = d[d["window"] == maxw].copy()

    # pick latest as_of_date
    row = d.iloc[-1]
    return row


def _alert_status(
    *,
    horizon_days: int,
    event_probability: float,
    cfg: FriendlySummaryConfig,
) -> str:
    if (int(horizon_days) in cfg.elevated_horizons) and (float(event_probability) >= cfg.elevated_prob_threshold):
        return "ELEVATED"
    return "NORMAL"


def build_friendly_risk_summary_from_artifacts(
    *,
    market: str,
    as_of_date: str,  # "YYYY-MM-DD" (used only for output filename + display)
    horizons: Iterable[int] = (5, 21, 63),
    cfg: FriendlySummaryConfig | None = None,
) -> Path:
    """
    Produces the recruiter-friendly CSV:

      as_of_date,horizon,event_probability,event_rate_rolling,calibration_gap,alert_status

    We derive values from the *latest artifact dir per horizon* using the rolling drift table:
      event_probability  := roll_p_mean (for a chosen window)
      event_rate_rolling := roll_event_rate
      calibration_gap    := event_probability - event_rate_rolling
    """
    cfg = cfg or FriendlySummaryConfig(market=market)

    preferred = cfg.preferred_windows_by_horizon or {5: 126, 21: 252, 63: 504}

    artifact_root = Path(cfg.artifact_root)
    out_dir = Path(cfg.out_dir)
    _ensure_dir(out_dir)

    rows: list[dict] = []
    for h in horizons:
        h = int(h)
        target_name = f"fwd_ret_{h}d_log"

        art_dir = _find_latest_artifact_dir(
            artifact_root=artifact_root,
            market=market,
            target_name=target_name,
            horizon_days=h,
        )

        drift = _read_rolling_drift(art_dir)
        row = _pick_best_row_for_horizon(drift, horizon_days=h, preferred_window=int(preferred.get(h, 0)))

        p = float(row["roll_p_mean"])
        r = float(row["roll_event_rate"])
        gap = p - r

        rows.append(
            {
                "as_of_date": str(as_of_date),
                "horizon": f"{h}d",
                "event_probability": round(p, 6),
                "event_rate_rolling": round(r, 6),
                "calibration_gap": round(gap, 6),
                "alert_status": _alert_status(horizon_days=h, event_probability=p, cfg=cfg),
            }
        )

    df = pd.DataFrame(rows, columns=[
        "as_of_date",
        "horizon",
        "event_probability",
        "event_rate_rolling",
        "calibration_gap",
        "alert_status",
    ])

    out_path = out_dir / f"friendly_risk_summary_{market}_{as_of_date}.csv"
    df.to_csv(out_path, index=False)
    return out_path
