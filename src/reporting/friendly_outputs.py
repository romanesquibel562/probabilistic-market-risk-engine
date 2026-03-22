# src/reporting/friendly_outputs.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd


EventRule = Literal["sigma", "quantile"]


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

    # Which event families to publish
    event_rules: tuple[EventRule, ...] = ("sigma", "quantile")

    # Simple, explainable alert rule (tweak later)
    elevated_horizons: tuple[int, ...] = (63,)
    elevated_prob_threshold: float = 0.25


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_event_family_from_dirname(dir_name: str) -> str:
    """
    Artifact dir examples:
      eventlogit_sigma_s1.25_topk_SPY_fwd_ret_5d_log_h5d_YYYYMMDD_HHMMSS
      eventlogit_quantile_q10_topk_SPY_fwd_ret_5d_log_h5d_YYYYMMDD_HHMMSS

    Return a friendly event_family string, with correct parsing (no "sigma" token captured as the parameter).
    """
    name = str(dir_name)
    parts = name.split("_")

    if name.startswith("eventlogit_sigma_"):
        # IMPORTANT: avoid accidentally capturing the literal token "sigma"
        # We only accept tokens like s1.25 (must be 's' + digit next)
        s_tag = next(
            (p for p in parts if p.startswith("s") and len(p) > 1 and p[1].isdigit()),
            None,
        )
        if s_tag:
            return f"downside_sigma_{s_tag}"
        return "downside_sigma"

    if name.startswith("eventlogit_quantile_"):
        # Accept tokens like q10
        q_tag = next(
            (p for p in parts if p.startswith("q") and len(p) > 1 and p[1:].isdigit()),
            None,
        )
        if q_tag:
            return f"tail_{q_tag}"
        return "tail_q"

    return "event_unknown"


def _find_latest_artifact_dir(
    *,
    artifact_root: Path,
    market: str,
    target_name: str,
    horizon_days: int,
    event_rule: EventRule | None = None,
) -> Path:
    """
    Finds the most recently modified artifact directory for a given (market, target, horizon[, rule]).
    Expected directory naming:
      artifacts/models/eventlogit_{rule}_..._{MARKET}_{TARGET}_h{H}d_YYYYMMDD_HHMMSS
    """
    if not artifact_root.exists():
        raise FileNotFoundError(f"artifact_root not found: {artifact_root.as_posix()}")

    key = f"_{market}_{target_name}_h{int(horizon_days)}d_"

    candidates: list[Path] = []
    for p in artifact_root.iterdir():
        if not p.is_dir():
            continue
        if key not in p.name:
            continue
        if event_rule is not None:
            # strict prefix match to avoid mixing families
            if not p.name.startswith(f"eventlogit_{event_rule}_"):
                continue
        candidates.append(p)

    if not candidates:
        extra = f" and event_rule={event_rule}" if event_rule else ""
        raise FileNotFoundError(
            f"No artifact dirs found for market={market} target={target_name} h={horizon_days}{extra} under {artifact_root.as_posix()}.\n"
            f"Expected a folder name containing: {key}"
        )

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_rolling_drift(artifact_dir: Path) -> pd.DataFrame:
    """
    Expects rolling drift CSV inside the artifact directory.
    """
    candidates = [
        artifact_dir / "rolling_calibration_drift.csv",
        artifact_dir / "rolling_calibration.csv",
        artifact_dir / "calibration_drift.csv",
        artifact_dir / "rolling_drift.csv",
    ]
    for fp in candidates:
        if fp.exists():
            return pd.read_csv(fp)

    raise FileNotFoundError(
        "Could not find rolling calibration drift CSV in artifact dir:\n"
        f"  {artifact_dir.as_posix()}\n"
        "Looked for one of:\n"
        + "\n".join([f"  - {c.name}" for c in candidates])
        + "\n\n"
        "Fix: make sure risk_event_logistic writes the drift table to 'rolling_calibration_drift.csv'."
    )


def _pick_best_row_for_horizon(
    drift: pd.DataFrame,
    *,
    preferred_window: int,
) -> pd.Series:
    """
    Picks the row we use to publish:
      - label should start with 'test_prod'
      - choose preferred_window if available, otherwise choose the largest window available
      - pick latest as_of_date
    """
    if drift.empty:
        raise ValueError("rolling drift df is empty")

    d = drift.copy()
    d["as_of_date"] = pd.to_datetime(d["as_of_date"], errors="coerce")
    d = d.dropna(subset=["as_of_date"]).sort_values("as_of_date").reset_index(drop=True)

    d = d[d["label"].astype(str).str.startswith("test_prod")].copy()
    if d.empty:
        raise ValueError("rolling drift has no rows with label starting with 'test_prod'")

    if "window" not in d.columns:
        raise ValueError("rolling drift is missing required column: window")

    d["window"] = pd.to_numeric(d["window"], errors="coerce").astype("Int64")
    if preferred_window and (d["window"] == preferred_window).any():
        d = d[d["window"] == preferred_window].copy()
    else:
        maxw = int(d["window"].dropna().max())
        d = d[d["window"] == maxw].copy()

    return d.iloc[-1]


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
    Produces the recruiter-friendly CSV with BOTH event families:

      as_of_date,market,event_family,event_rule,horizon,event_probability,event_rate_rolling,calibration_gap,alert_status,artifact_dir

    We derive values from the *latest artifact dir per horizon per event_rule* using the rolling drift table:
      event_probability   := roll_p_mean (for a chosen window)
      event_rate_rolling  := roll_event_rate
      calibration_gap     := event_probability - event_rate_rolling
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
        pref_window = int(preferred.get(h, 0))

        for rule in cfg.event_rules:
            art_dir = _find_latest_artifact_dir(
                artifact_root=artifact_root,
                market=market,
                target_name=target_name,
                horizon_days=h,
                event_rule=rule,
            )

            drift = _read_rolling_drift(art_dir)
            row = _pick_best_row_for_horizon(drift, preferred_window=pref_window)

            p = float(row["roll_p_mean"])
            r = float(row["roll_event_rate"])
            gap = p - r

            rows.append(
                {
                    "as_of_date": str(as_of_date),
                    "market": str(market),
                    "event_family": _parse_event_family_from_dirname(art_dir.name),
                    "event_rule": str(rule),
                    "horizon": f"{h}d",
                    "event_probability": round(p, 6),
                    "event_rate_rolling": round(r, 6),
                    "calibration_gap": round(gap, 6),
                    "alert_status": _alert_status(horizon_days=h, event_probability=p, cfg=cfg),
                    "artifact_dir": art_dir.as_posix(),
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "as_of_date",
            "market",
            "event_family",
            "event_rule",
            "horizon",
            "event_probability",
            "event_rate_rolling",
            "calibration_gap",
            "alert_status",
            "artifact_dir",
        ],
    )

    out_path = out_dir / f"friendly_risk_summary_{market}_{as_of_date}.csv"
    df.to_csv(out_path, index=False)
    return out_path
