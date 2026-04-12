# src/models/common/risk_event_logistic.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

from src.core.config import settings
from src.training.training_matrix import build_training_matrix_asof


# ----------------------------
# Config
# ----------------------------

CalibSplit = Literal["tail", "interleaved"]
EventRule = Literal["sigma", "quantile"]
RankMethod = Literal["score", "prob"]


@dataclass
class EventLogitConfig:
    market: str

    # leakage-safe read lock
    as_of_ts: datetime | None = None

    # target
    horizon_days: int = 5
    target_name: str = "fwd_ret_5d_log"
    target_version: str | None = None

    # features
    feature_version: str | None = None

    # windows are “rows” in your output (not literal days)
    test_window_days: int = 90
    calib_window_days: int = 90

    # event definition
    event_rule: EventRule = "sigma"
    sigma_mult: float = 1.25
    event_quantile: float = 0.10

    # convenience
    auto_tune_defaults: bool = True

    # calibration
    calib_split: CalibSplit = "interleaved"
    allow_isotonic: bool = True

    # base-rate alignment
    apply_prior_correction: bool = True
    prior_anchor: Literal["fit", "calib"] = "calib"

    # ranking selection (kept, but we force Option A below)
    choose_rank_method: bool = True
    rank_metric_k: float = 0.10

    # fit capping
    fit_max_rows: int | None = None
    fit_tail_only: bool = True
    debug_splits: bool = False

    # artifacts
    out_dir: str = "artifacts/models"

    # calibration guards
    spread_min: float = 0.05
    p_std_min: float | None = None
    unique_ratio_min: float = 0.05
    min_levels: int = 12
    max_auc_drop: float = 0.02
    max_ap_drop: float = 0.02

    # model
    class_weight: str | None = None


# ----------------------------
# Utilities
# ----------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _logit(p: np.ndarray) -> np.ndarray:
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _has_both_classes(y: np.ndarray) -> bool:
    y = np.asarray(y, dtype=int)
    return len(np.unique(y)) == 2


def _p_std_min_for_horizon(h: int) -> float:
    # horizon-aware minimum spread/variance guard
    if h >= 63:
        return 0.01
    if h >= 21:
        return 0.02
    return 0.035


def _auto_tune_defaults(cfg: EventLogitConfig) -> EventLogitConfig:
    """
    Horizon + rule-specific defaults.
    Key principles:
      - 5d: NEVER use class_weight="balanced" (it inflates base-rate; you already fixed this)
      - quantile events: avoid class_weight="balanced" (it can badly distort probabilities)
      - isotonic: only allow for 5d
      - p_std_min: looser for rarer events/horizons
    """
    if not cfg.auto_tune_defaults:
        return cfg

    cfg = EventLogitConfig(**cfg.__dict__)  # shallow copy

    # ---- universal: quantile events should not use balanced weights
    if cfg.event_rule == "quantile":
        cfg.class_weight = None

    if cfg.horizon_days == 5:
        cfg.class_weight = None  # critical for 5d (sigma + quantile)

        cfg.allow_isotonic = True
        cfg.test_window_days = 90
        cfg.calib_window_days = 252

        if cfg.event_rule == "sigma":
            cfg.p_std_min = 0.035
            cfg.spread_min = 0.06
            cfg.unique_ratio_min = 0.15
            cfg.min_levels = 12
            cfg.max_auc_drop = 0.02
            cfg.max_ap_drop = 0.02
        else:
            cfg.p_std_min = 0.02
            cfg.spread_min = 0.04
            cfg.unique_ratio_min = 0.10
            cfg.min_levels = 12
            cfg.max_auc_drop = 0.02
            cfg.max_ap_drop = 0.02

    elif cfg.horizon_days == 21:
        cfg.allow_isotonic = False
        cfg.test_window_days = 252
        cfg.calib_window_days = 252

        cfg.spread_min = 0.05
        cfg.unique_ratio_min = 0.12
        cfg.max_auc_drop = 0.02
        cfg.max_ap_drop = 0.02
        cfg.p_std_min = 0.02
        cfg.min_levels = 10

        # balanced is OK for sigma; quantile already forced to None above
        if cfg.event_rule == "sigma":
            cfg.class_weight = "balanced"

    elif cfg.horizon_days == 63:
        cfg.allow_isotonic = False
        cfg.apply_prior_correction = True
        cfg.test_window_days = 504
        cfg.calib_window_days = 504

        cfg.p_std_min = 0.01
        cfg.min_levels = 12
        cfg.spread_min = 0.04
        cfg.unique_ratio_min = 0.08
        cfg.max_auc_drop = 0.03
        cfg.max_ap_drop = 0.03

        # balanced is OK for sigma; quantile already forced to None above
        if cfg.event_rule == "sigma":
            cfg.class_weight = "balanced"

    return cfg


def _platt_fit(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    s = np.asarray(s, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(s, y)
    a = float(lr.coef_.ravel()[0])
    b = float(lr.intercept_.ravel()[0])
    return a, b


def _platt_predict(s: np.ndarray, a: float, b: float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    return _sigmoid(a * s + b)


def _pick_vol_col(df: pd.DataFrame, h: int) -> str:
    candidates = [f"rv_{h}d", f"rv_{h}", "rv_63d", "rv_63", "rv_21d", "rv_21", "rv_5d", "rv_5"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No realized vol column found. Tried: {candidates}")


def _solve_intercept_shift(p: np.ndarray, target_rate: float) -> float:
    eps = 1e-9
    target_rate = float(np.clip(target_rate, eps, 1 - eps))
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    z = _logit(p)

    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pm = float(_sigmoid(z + mid).mean())
        if pm < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _apply_intercept_shift(p: np.ndarray, shift: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return _sigmoid(_logit(p) + float(shift))


def _reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.digitize(p, edges, right=True) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_id == b
        if not mask.any():
            continue
        pb = p[mask]
        yb = y[mask]
        p_mean = float(pb.mean())
        y_rate = float(yb.mean())
        rows.append(
            {
                "bin": int(b),
                "p_min": float(edges[b]),
                "p_max": float(edges[b + 1]),
                "n": int(mask.sum()),
                "p_mean": p_mean,
                "y_rate": y_rate,
                "gap_y_minus_p": float(y_rate - p_mean),
            }
        )
    return pd.DataFrame(rows)


def _rolling_calibration(
    df: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    *,
    window: int = 63,
    min_periods: int | None = None,
    label: str = "test_prod",
) -> pd.DataFrame:
    if min_periods is None:
        min_periods = max(10, int(window * 0.5))

    if len(df) != len(y) or len(df) != len(p):
        raise ValueError("df, y, p must have same length for rolling calibration.")

    tmp = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(df["as_of_date"], errors="coerce"),
            "y": np.asarray(y, dtype=int),
            "p": np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9),
        }
    )
    tmp = tmp.dropna(subset=["as_of_date"]).sort_values("as_of_date").reset_index(drop=True)

    def _brier_window(yw: np.ndarray, pw: np.ndarray) -> float:
        return float(brier_score_loss(yw, pw))

    def _logloss_window(yw: np.ndarray, pw: np.ndarray) -> float:
        return float(log_loss(yw, pw, labels=[0, 1]))

    out_rows = []
    n = len(tmp)
    for i in range(n):
        j0 = max(0, i - window + 1)
        yw = tmp["y"].iloc[j0 : i + 1].to_numpy()
        pw = tmp["p"].iloc[j0 : i + 1].to_numpy()

        if (i + 1 - j0) < min_periods:
            out_rows.append(
                {
                    "as_of_date": tmp["as_of_date"].iloc[i],
                    "label": label,
                    "window": int(window),
                    "n_window": int(i + 1 - j0),
                    "roll_event_rate": float("nan"),
                    "roll_p_mean": float("nan"),
                    "roll_brier": float("nan"),
                    "roll_logloss": float("nan"),
                }
            )
            continue

        out_rows.append(
            {
                "as_of_date": tmp["as_of_date"].iloc[i],
                "label": label,
                "window": int(window),
                "n_window": int(i + 1 - j0),
                "roll_event_rate": float(yw.mean()),
                "roll_p_mean": float(pw.mean()),
                "roll_brier": _brier_window(yw, pw),
                "roll_logloss": _logloss_window(yw, pw),
            }
        )

    return pd.DataFrame(out_rows)


# ----------------------------
# Event definition
# ----------------------------

@dataclass
class EventDef:
    rule: EventRule
    quantile_cut: float | None = None
    vol_col: str | None = None


def _validate_target_pair(target_name: str, horizon_days: int) -> None:
    expected = f"fwd_ret_{horizon_days}d_log"
    if target_name.startswith("fwd_ret_") and target_name.endswith("_log") and target_name != expected:
        raise ValueError(
            f"target_name='{target_name}' does not match horizon_days={horizon_days}. "
            f"Expected '{expected}'."
        )


def _fit_event_definition(fit_df: pd.DataFrame, *, cfg: EventLogitConfig) -> EventDef:
    if cfg.fit_max_rows is not None:
        before = len(fit_df)
        cap = int(cfg.fit_max_rows)
        if before > cap:
            fit_df = fit_df.tail(cap).copy() if cfg.fit_tail_only else fit_df.sample(cap, random_state=0).copy()
        after = len(fit_df)
        if cfg.debug_splits:
            print(f"[fit_cap:eventdef] before={before} cap={cap} after={after} tail_only={cfg.fit_tail_only}")

    if cfg.event_rule == "sigma":
        vol_col = _pick_vol_col(fit_df, cfg.horizon_days)
        return EventDef(rule="sigma", vol_col=vol_col)

    if cfg.event_rule == "quantile":
        y_fwd = fit_df["target_value"].astype(float).values
        cut = float(np.nanquantile(y_fwd, float(cfg.event_quantile)))
        return EventDef(rule="quantile", quantile_cut=cut)

    raise ValueError(f"Unsupported event_rule={cfg.event_rule}")


def _make_event_labels(
    df: pd.DataFrame,
    *,
    cfg: EventLogitConfig,
    evdef: EventDef,
) -> tuple[np.ndarray, pd.Series]:
    y_fwd = df["target_value"].astype(float).values

    if evdef.rule == "sigma":
        assert evdef.vol_col is not None
        vol = df[evdef.vol_col].astype(float).values
        thresh = -float(cfg.sigma_mult) * vol
        y_event = (y_fwd <= thresh).astype(int)
        return y_event, pd.Series(thresh, index=df.index, name="threshold")

    if evdef.rule == "quantile":
        assert evdef.quantile_cut is not None
        cut = float(evdef.quantile_cut)
        y_event = (y_fwd <= cut).astype(int)
        return y_event, pd.Series(cut, index=df.index, name="threshold")

    raise ValueError(f"Unsupported event_rule={evdef.rule}")


def _split_fit_calib_test(
    df: pd.DataFrame,
    *,
    test_n: int,
    calib_n: int,
    mode: CalibSplit,
    fit_max_rows: int | None = None,
    fit_tail_only: bool = True,
    debug_splits: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) < (test_n + calib_n + 50):
        raise ValueError(f"Not enough rows for split. rows={len(df)} needs >= {test_n+calib_n+50}")

    df = df.sort_values("as_of_date").reset_index(drop=True)

    tail = df.iloc[-(test_n + calib_n):].copy()
    fit = df.iloc[: -(test_n + calib_n)].copy()
    if fit.empty:
        raise ValueError("Fit split is empty; increase history or shrink windows.")

    if fit_max_rows is not None:
        before = len(fit)
        cap = int(fit_max_rows)
        if before > cap:
            fit = fit.tail(cap).copy() if fit_tail_only else fit.sample(cap, random_state=0).copy()
        after = len(fit)
        if debug_splits:
            print(f"[fit_cap] before={before} cap={cap} after={after} tail_only={fit_tail_only}")

    if mode == "tail":
        calib = tail.iloc[:calib_n].copy()
        test = tail.iloc[calib_n:].copy()
        return fit, calib, test

    if mode == "interleaved":
        tail = tail.reset_index(drop=True)
        mask = np.zeros(len(tail), dtype=bool)
        mask[::2] = True
        calib = tail[mask].copy()
        test = tail[~mask].copy()

        if len(calib) > calib_n:
            calib = calib.iloc[-calib_n:].copy()
        if len(test) > test_n:
            test = test.iloc[-test_n:].copy()

        if len(calib) < int(0.8 * calib_n) or len(test) < int(0.8 * test_n):
            calib = tail.iloc[:calib_n].copy()
            test = tail.iloc[calib_n:].copy()

        return fit, calib, test

    raise ValueError(f"Unsupported calib_split={mode}")


def _prob_stats(p: np.ndarray) -> str:
    p = np.asarray(p, dtype=float)
    q05, q50, q95 = np.quantile(p, [0.05, 0.50, 0.95])
    q99 = np.quantile(p, 0.99)
    return (
        f"min={p.min():.6f} p05={q05:.6f} p50={q50:.6f} p95={q95:.6f} max={p.max():.6f} "
        f"| frac>0.9={(p>0.9).mean():.3f} frac<0.1={(p<0.1).mean():.3f}"
    )


def _metrics(tag: str, y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p_clip = np.clip(p, 1e-9, 1 - 1e-9)
    has_two = len(np.unique(y)) > 1
    return {
        "tag": tag,
        "event_rate": float(y.mean()) if len(y) else float("nan"),
        "log_loss": float(log_loss(y, p_clip, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p_clip)),
        "auc": float(roc_auc_score(y, p_clip)) if has_two else float("nan"),
        "ap": float(average_precision_score(y, p_clip)) if has_two else float("nan"),
        "p_mean": float(p_clip.mean()),
        "p_p95": float(np.quantile(p_clip, 0.95)),
        "p_p99": float(np.quantile(p_clip, 0.99)),
    }


def _topk_alerts(tag_prefix: str, y: np.ndarray, score: np.ndarray, ks=(0.05, 0.10)) -> list[dict]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    out = []
    n = len(score)
    if n == 0:
        return out

    for k in ks:
        k = float(k)
        m = max(1, int(round(k * n)))
        idx = np.argsort(-score)[:m]
        alert = np.zeros(n, dtype=int)
        alert[idx] = 1

        tp = float(((alert == 1) & (y == 1)).sum())
        fp = float(((alert == 1) & (y == 0)).sum())
        tn = float(((alert == 0) & (y == 0)).sum())
        fn = float(((alert == 0) & (y == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        out.append(
            {
                "tag": f"{tag_prefix}_top{int(k*100)}%",
                "k": k,
                "alert_rate": float(alert.mean()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return out


def _select_best_calibration(
    *,
    y_ref: np.ndarray,
    p_raw: np.ndarray,
    p_sig: np.ndarray | None,
    p_iso: np.ndarray | None,
    min_improve: float = 1e-6,
    spread_min: float = 0.05,
    p_std_min: float = 0.02,
    unique_ratio_min: float = 0.05,
    min_levels: int = 12,
    max_auc_drop: float = 0.02,
    max_ap_drop: float = 0.02,
    label: str = "calib",
) -> tuple[str, dict]:
    """
    Generic calibration selector that can run on CALIB or FIT when CALIB is inverted.
    """
    y_ref = np.asarray(y_ref, dtype=int)

    def clip(p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)

    def score_ref(p: np.ndarray) -> dict:
        p = clip(p)
        b = float(brier_score_loss(y_ref, p))
        ll = float(log_loss(y_ref, p, labels=[0, 1]))

        q05, q95 = np.quantile(p, [0.05, 0.95])
        spread = float(q95 - q05)

        if _has_both_classes(y_ref):
            auc = float(roc_auc_score(y_ref, p))
            ap = float(average_precision_score(y_ref, p))
        else:
            auc = float("nan")
            ap = float("nan")

        p_std = float(np.std(p))
        uniq = np.unique(np.round(p, 6))
        unique_ratio = float(len(uniq) / max(1, len(p)))
        n_levels = int(len(uniq))

        return {
            "brier": b,
            "log_loss": ll,
            "spread_p95_p05": spread,
            "p_std": p_std,
            "unique_ratio": unique_ratio,
            "n_levels": n_levels,
            "auc": auc,
            "ap": ap,
        }

    scores: dict[str, dict[str, float] | dict] = {}
    scores["raw"] = score_ref(p_raw)
    raw = scores["raw"]

    candidates: list[str] = ["raw"]
    if p_sig is not None:
        scores["sigmoid"] = score_ref(p_sig)
        candidates.append("sigmoid")
    if p_iso is not None:
        scores["isotonic"] = score_ref(p_iso)
        candidates.append("isotonic")

    rejected: dict[str, str] = {}
    viable: list[str] = ["raw"]

    for name in candidates:
        if name == "raw":
            continue
        s = scores[name]  # type: ignore[assignment]

        # Reject inverted candidates on this reference split (only meaningful if both classes exist)
        if _has_both_classes(y_ref) and np.isfinite(s.get("auc", float("nan"))) and s["auc"] < 0.5:
            rejected[name] = "auc<0.500 (inverted)"
            continue

        if s["spread_p95_p05"] < spread_min:
            rejected[name] = f"spread<{spread_min:.3f}"
            continue
        if s.get("p_std", 1.0) < p_std_min:
            rejected[name] = f"p_std<{p_std_min:.3f}"
            continue

        if name == "isotonic":
            if s["n_levels"] < min_levels:
                rejected[name] = f"n_levels<{min_levels}"
                continue
        else:
            if s.get("unique_ratio", 1.0) < unique_ratio_min:
                rejected[name] = f"unique_ratio<{unique_ratio_min:.3f}"
                continue

        if np.isfinite(raw["auc"]) and np.isfinite(s["auc"]):
            if (raw["auc"] - s["auc"]) > max_auc_drop:
                rejected[name] = f"auc_drop>{max_auc_drop:.3f}"
                continue
        if np.isfinite(raw["ap"]) and np.isfinite(s["ap"]):
            if (raw["ap"] - s["ap"]) > max_ap_drop:
                rejected[name] = f"ap_drop>{max_ap_drop:.3f}"
                continue

        viable.append(name)

    best_name = sorted(viable, key=lambda n: (scores[n]["brier"], scores[n]["log_loss"]))[0]  # type: ignore[index]

    if best_name != "raw":
        if (raw["brier"] - scores[best_name]["brier"]) < float(min_improve):  # type: ignore[index]
            best_name = "raw"

    scores["rejected"] = rejected
    scores["chosen"] = {"name": best_name}
    scores["selection_label"] = {"label": label}
    return best_name, scores  # type: ignore[return-value]


def _build_prod_probs_from_choice(
    *,
    chosen: str,
    p_raw: np.ndarray,
    p_sig: np.ndarray | None,
    p_iso: np.ndarray | None,
) -> np.ndarray:
    if chosen == "sigmoid" and p_sig is not None:
        return p_sig
    if chosen == "isotonic" and p_iso is not None:
        return p_iso
    return p_raw


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "market",
        "as_of_date",
        "forward_date",
        "target_value",
        "target_name",
        "horizon_days",
        "target_version",
        "feature_version",
        "available_time",
        "computed_at",
        "run_id",
        "source",
        "series_id",
        "value",
        "ingested_at",
    }
    cols = [c for c in df.columns if c not in exclude]
    if not cols:
        raise ValueError("No feature columns left after exclude list.")
    return cols


# ----------------------------
# Main runner
# ----------------------------

def run_step6_event_logistic(cfg: EventLogitConfig) -> None:
    cfg = _auto_tune_defaults(cfg)

    if cfg.feature_version is None:
        cfg.feature_version = settings.DEFAULT_FEATURE_VERSION
    if cfg.target_version is None:
        cfg.target_version = settings.DEFAULT_TARGET_VERSION

    _validate_target_pair(cfg.target_name, cfg.horizon_days)

    if cfg.as_of_ts is None:
        cfg.as_of_ts = _utc_now()
    else:
        if cfg.as_of_ts.tzinfo is None:
            cfg.as_of_ts = cfg.as_of_ts.replace(tzinfo=timezone.utc)
        else:
            cfg.as_of_ts = cfg.as_of_ts.astimezone(timezone.utc)

    print("\n=== STEP 6.1: Logistic Risk-Event + Calibration Guard + Top-K ===")
    event_params = f"sigma_mult={cfg.sigma_mult}" if cfg.event_rule == "sigma" else f"q={cfg.event_quantile:.2f}"

    print(
        "Config: "
        f"market={cfg.market} target={cfg.target_name} h={cfg.horizon_days} "
        f"rule={cfg.event_rule} {event_params} "
        f"test={cfg.test_window_days} calib={cfg.calib_window_days} "
        f"calib_split={cfg.calib_split} allow_iso={cfg.allow_isotonic} "
        f"as_of_ts={cfg.as_of_ts.isoformat()}"
    )

    # 1) build leakage-safe matrix
    df = build_training_matrix_asof(
        market=cfg.market,
        as_of_ts=cfg.as_of_ts,
        feature_version=cfg.feature_version,
        target_name=cfg.target_name,
        horizon_days=cfg.horizon_days,
        target_version=cfg.target_version,
        dropna_features=True,
    )
    if df.empty:
        raise ValueError("Training matrix is empty after joins/dropna.")

    # 2) split
    fit_df, calib_df, test_df = _split_fit_calib_test(
        df,
        test_n=int(cfg.test_window_days),
        calib_n=int(cfg.calib_window_days),
        mode=cfg.calib_split,
        fit_max_rows=cfg.fit_max_rows,
        fit_tail_only=cfg.fit_tail_only,
        debug_splits=cfg.debug_splits,
    )
    print(f"Rows total: {len(df)} | Fit: {len(fit_df)} | Calib: {len(calib_df)} | Test: {len(test_df)}")

    # 3) event definition on FIT only
    evdef = _fit_event_definition(fit_df, cfg=cfg)

    # 4) labels
    y_fit, _ = _make_event_labels(fit_df, cfg=cfg, evdef=evdef)
    y_calib, _ = _make_event_labels(calib_df, cfg=cfg, evdef=evdef)
    y_test, _ = _make_event_labels(test_df, cfg=cfg, evdef=evdef)

    if cfg.event_rule == "sigma":
        assert evdef.vol_col is not None
        print(
            f"Event rule: y_fwd <= -{cfg.sigma_mult:.3f} * vol_asof(h={cfg.horizon_days}d)  [vol_col={evdef.vol_col}]"
        )
    else:
        assert evdef.quantile_cut is not None
        print(f"Event rule: y_fwd <= quantile_cut(q={cfg.event_quantile:.2f}) = {evdef.quantile_cut:.6f} (fit-only)")

    print(f"Event rates: fit={y_fit.mean():.3f}, calib={y_calib.mean():.3f}, test={y_test.mean():.3f}")

    # 5) model
    feature_cols = _select_feature_columns(df)

    X_fit = fit_df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float).values
    X_calib = calib_df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float).values
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float).values

    base = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight=cfg.class_weight,
    )
    base.fit(X_fit, y_fit)

    s_fit = base.decision_function(X_fit)
    s_calib = base.decision_function(X_calib)
    s_test = base.decision_function(X_test)

    p_fit_raw = base.predict_proba(X_fit)[:, 1]
    p_calib_raw = base.predict_proba(X_calib)[:, 1]
    p_test_raw = base.predict_proba(X_test)[:, 1]

    # ---------------------------
    # Direction: decide on FIT only
    # ---------------------------
    auc_fit_raw = float("nan")
    if _has_both_classes(y_fit):
        auc_fit_raw = float(roc_auc_score(y_fit, np.clip(p_fit_raw, 1e-9, 1 - 1e-9)))

    flipped = False
    if np.isfinite(auc_fit_raw) and auc_fit_raw < 0.5:
        flipped = True
        print(f"WARNING: fit AUC indicates inversion ({auc_fit_raw:.4f}); flipping direction.")
        s_fit = -s_fit
        s_calib = -s_calib
        s_test = -s_test
        p_fit_raw = 1.0 - p_fit_raw
        p_calib_raw = 1.0 - p_calib_raw
        p_test_raw = 1.0 - p_test_raw

    print(f"[direction] auc_fit_raw={auc_fit_raw:.4f} flipped={flipped}")
    print(f"p_calib_raw: {_prob_stats(p_calib_raw)}")
    print(f"p_test_raw:  {_prob_stats(p_test_raw)}")
    print(f"Class rates: y_fit={y_fit.mean():.3f} | y_calib={y_calib.mean():.3f} | y_test={y_test.mean():.3f}")

    # ---------------------------
    # CALIB inversion diagnostic + safe fallback
    # ---------------------------
    auc_calib_raw = float("nan")
    calib_inverted = False
    if _has_both_classes(y_calib):
        auc_calib_raw = float(roc_auc_score(y_calib, np.clip(p_calib_raw, 1e-9, 1 - 1e-9)))
        if np.isfinite(auc_calib_raw) and auc_calib_raw < 0.5:
            calib_inverted = True
            print(
                f"WARNING: calib AUC is inverted (auc_calib_raw={auc_calib_raw:.4f}). "
                "Disabling CALIB-based calibration selection; using FIT-based selection for stream choice."
            )

    can_calibrate = _has_both_classes(y_calib)

    p_calib_iso: np.ndarray | None = None
    p_test_iso: np.ndarray | None = None
    p_fit_iso: np.ndarray | None = None

    p_fit_sig: np.ndarray | None = None
    p_calib_sig: np.ndarray | None = None
    p_test_sig: np.ndarray | None = None

    a: float | None = None
    b: float | None = None
    iso: IsotonicRegression | None = None

    if not can_calibrate:
        print("WARNING: calib split has a single class; skipping isotonic/sigmoid calibration.")
    else:
        # ---- Isotonic (only for 5d)
        if cfg.allow_isotonic:
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                # IMPORTANT: fit on score (monotonic in score)
                iso.fit(s_calib, y_calib)
                p_calib_iso = iso.predict(s_calib)
                p_test_iso = iso.predict(s_test)
                p_fit_iso = iso.predict(s_fit)
            except Exception as e:
                print(f"Calibration isotonic skipped due to error: {e}")
                iso = None
                p_calib_iso = p_test_iso = p_fit_iso = None
        else:
            print("Isotonic disabled by config (recommended for h>=21).")

        # ---- Platt / Sigmoid (fit on CALIB)
        try:
            a, b = _platt_fit(s_calib, y_calib)

            # If Platt stream is inverted on FIT, fix by flipping scores fed into Platt
            auc_fit_sig = float("nan")
            if _has_both_classes(y_fit):
                p_fit_sig_tmp = _platt_predict(s_fit, a, b)
                auc_fit_sig = float(roc_auc_score(y_fit, np.clip(p_fit_sig_tmp, 1e-9, 1 - 1e-9)))

            if np.isfinite(auc_fit_sig) and auc_fit_sig < 0.5:
                print(
                    f"WARNING: Platt inverted on FIT (auc={auc_fit_sig:.4f}); "
                    "flipping scores for sigmoid stream."
                )
                p_fit_sig = _platt_predict(-s_fit, a, b)
                p_calib_sig = _platt_predict(-s_calib, a, b)
                p_test_sig = _platt_predict(-s_test, a, b)
            else:
                p_fit_sig = _platt_predict(s_fit, a, b)
                p_calib_sig = _platt_predict(s_calib, a, b)
                p_test_sig = _platt_predict(s_test, a, b)

            print(f"Platt params: a={a:.6f} b={b:.6f} | auc_fit_sig={auc_fit_sig:.4f}")

        except Exception as e:
            print(f"Calibration sigmoid skipped due to error: {e}")
            p_fit_sig = p_calib_sig = p_test_sig = None
            a = b = None

    # ---------------------------
    # Stream selection (PERFECTED):
    #   - Normal: choose stream based on CALIB metrics/guards
    #   - If CALIB inverted: choose stream based on FIT metrics/guards (stable fallback)
    # ---------------------------
    chosen = "raw"
    calib_scores: dict = {"chosen": {"name": "raw"}, "rejected": {}, "selection_label": {"label": "none"}}

    p_std_min = cfg.p_std_min if cfg.p_std_min is not None else _p_std_min_for_horizon(cfg.horizon_days)

    if can_calibrate:
        if calib_inverted:
            chosen, calib_scores = _select_best_calibration(
                y_ref=y_fit,
                p_raw=p_fit_raw,
                p_sig=p_fit_sig,
                p_iso=p_fit_iso,
                min_improve=1e-6,
                spread_min=cfg.spread_min,
                p_std_min=p_std_min,
                unique_ratio_min=cfg.unique_ratio_min,
                min_levels=cfg.min_levels,
                max_auc_drop=cfg.max_auc_drop,
                max_ap_drop=cfg.max_ap_drop,
                label="fit_fallback_due_to_calib_inversion",
            )
        else:
            chosen, calib_scores = _select_best_calibration(
                y_ref=y_calib,
                p_raw=p_calib_raw,
                p_sig=p_calib_sig,
                p_iso=p_calib_iso,
                min_improve=1e-6,
                spread_min=cfg.spread_min,
                p_std_min=p_std_min,
                unique_ratio_min=cfg.unique_ratio_min,
                min_levels=cfg.min_levels,
                max_auc_drop=cfg.max_auc_drop,
                max_ap_drop=cfg.max_ap_drop,
                label="calib",
            )

    p_fit_prod = _build_prod_probs_from_choice(chosen=chosen, p_raw=p_fit_raw, p_sig=p_fit_sig, p_iso=p_fit_iso)
    p_calib_prod = _build_prod_probs_from_choice(chosen=chosen, p_raw=p_calib_raw, p_sig=p_calib_sig, p_iso=p_calib_iso)
    p_test_prod = _build_prod_probs_from_choice(chosen=chosen, p_raw=p_test_raw, p_sig=p_test_sig, p_iso=p_test_iso)

    print(f"Calibration selection: chosen={chosen} | scores={calib_scores}")

    platt_params = {"a": float(a), "b": float(b)} if (can_calibrate and a is not None and b is not None) else None

    # ---------------------------
    # Base-rate alignment (intercept-only correction)
    # ---------------------------
    prior_correction_meta = {"enabled": False}
    if cfg.apply_prior_correction:
        anchor = cfg.prior_anchor
        y_anchor = y_fit if anchor == "fit" else y_calib
        p_anchor = p_fit_prod if anchor == "fit" else p_calib_prod

        pi_anchor = float(np.mean(y_anchor))
        pbar_anchor = float(np.mean(p_anchor))
        shift = _solve_intercept_shift(p_anchor, pi_anchor)

        def _brier_ll(y, p):
            p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
            return float(brier_score_loss(y, p)), float(log_loss(y, p, labels=[0, 1]))

        b_before, ll_before = _brier_ll(y_anchor, p_anchor)
        p_anchor_after = _apply_intercept_shift(p_anchor, shift)
        b_after, ll_after = _brier_ll(y_anchor, p_anchor_after)

        shift_max = 6.0
        apply_shift = (ll_after <= ll_before + 1e-6) and (abs(shift) <= shift_max)

        prior_correction_meta = {
            "enabled": True,
            "anchor": anchor,
            "horizon_days": int(cfg.horizon_days),
            "chosen_stream": str(chosen),
            "pi_anchor": float(pi_anchor),
            "pbar_anchor": float(pbar_anchor),
            "shift_logit": float(shift),
            "anchor_brier_before": b_before,
            "anchor_logloss_before": ll_before,
            "anchor_brier_after": b_after,
            "anchor_logloss_after": ll_after,
            "applied": bool(apply_shift),
            "calib_mean_before": float(np.mean(p_calib_prod)),
            "test_mean_before": float(np.mean(p_test_prod)),
        }

        if apply_shift:
            p_fit_prod = _apply_intercept_shift(p_fit_prod, shift)
            p_calib_prod = _apply_intercept_shift(p_calib_prod, shift)
            p_test_prod = _apply_intercept_shift(p_test_prod, shift)

        prior_correction_meta.update(
            {
                "calib_mean_after": float(np.mean(p_calib_prod)),
                "test_mean_after": float(np.mean(p_test_prod)),
            }
        )

        print(
            f"[prior_align] h={cfg.horizon_days} anchor={anchor} chosen={chosen} applied={apply_shift} "
            f"pi={pi_anchor:.4f} pbar={pbar_anchor:.4f} shift={shift:+.4f} "
            f"calib_mean {prior_correction_meta['calib_mean_before']:.4f}->{prior_correction_meta['calib_mean_after']:.4f} "
            f"test_mean {prior_correction_meta['test_mean_before']:.4f}->{prior_correction_meta['test_mean_after']:.4f}"
        )

    # reliability
    rel_tables: dict[str, pd.DataFrame] = {}
    rel_tables["test_prod"] = _reliability_table(y_test, p_test_prod, n_bins=10)
    rel_tables["test_raw"] = _reliability_table(y_test, p_test_raw, n_bins=10)
    if p_test_sig is not None:
        rel_tables["test_sigmoid"] = _reliability_table(y_test, p_test_sig, n_bins=10)
    if p_test_iso is not None:
        rel_tables["test_isotonic"] = _reliability_table(y_test, p_test_iso, n_bins=10)

    rel_tables["calib_prod"] = _reliability_table(y_calib, p_calib_prod, n_bins=10)
    rel_tables["calib_raw"] = _reliability_table(y_calib, p_calib_raw, n_bins=10)
    if p_calib_sig is not None:
        rel_tables["calib_sigmoid"] = _reliability_table(y_calib, p_calib_sig, n_bins=10)
    if p_calib_iso is not None:
        rel_tables["calib_isotonic"] = _reliability_table(y_calib, p_calib_iso, n_bins=10)

    print("\n--- Reliability (TEST: prod) ---")
    print(rel_tables["test_prod"].to_string(index=False))

    print("\n--- Reliability (CALIB: prod) ---")
    print(rel_tables["calib_prod"].to_string(index=False))

    # rolling drift
    rolling_frames: list[pd.DataFrame] = []
    if cfg.horizon_days == 5:
        windows = [63, 126]
    elif cfg.horizon_days == 21:
        windows = [126, 252]
    else:
        windows = [252, 504]

    for w in windows:
        n_test = int(pd.to_datetime(test_df["as_of_date"], errors="coerce").notna().sum())
        mp = min(n_test, max(20, int(w * 0.5)))
        rolling_frames.append(_rolling_calibration(test_df, y_test, p_test_prod, window=w, min_periods=mp, label=f"test_prod_w{w}"))
        rolling_frames.append(_rolling_calibration(test_df, y_test, p_test_raw, window=w, min_periods=mp, label=f"test_raw_w{w}"))

    rolling_calib_df = pd.concat(rolling_frames, ignore_index=True)
    latest_rows = (
        rolling_calib_df.sort_values("as_of_date")
        .groupby("label", as_index=False)
        .tail(1)
        .sort_values("label")
    )

    print("\n--- Rolling Calibration Drift (latest per stream) ---")
    print(latest_rows.to_string(index=False))

    # ranking: always by score (Option A)
    rank_method: RankMethod = "score"
    rank_selection: dict = {
        "rank_method": "score",
        "rank_reason": "always_rank_by_score_option_A",
        "forced": True,
    }
    rank_score_test = s_test
    print(f"Rank method: chosen={rank_method} | {rank_selection}")

    # metrics
    rows = [
        _metrics("test_prod", y_test, p_test_prod),
        _metrics("test_raw", y_test, p_test_raw),
    ]
    if p_test_sig is not None:
        rows.append(_metrics("test_sigmoid", y_test, p_test_sig))
    if p_test_iso is not None:
        rows.append(_metrics("test_isotonic", y_test, p_test_iso))
    mdf = pd.DataFrame(rows).sort_values("brier", ascending=True)

    print("\n--- Test Probability Metrics (sorted by Brier, lower is better) ---")
    print(mdf.to_string(index=False))

    calib_rows = [
        _metrics("calib_prod", y_calib, p_calib_prod),
        _metrics("calib_raw", y_calib, p_calib_raw),
    ]
    if p_calib_sig is not None:
        calib_rows.append(_metrics("calib_sigmoid", y_calib, p_calib_sig))
    if p_calib_iso is not None:
        calib_rows.append(_metrics("calib_isotonic", y_calib, p_calib_iso))
    calib_mdf = pd.DataFrame(calib_rows).sort_values("brier", ascending=True)

    print("\n--- Calib Probability Metrics (sorted by Brier, lower is better) ---")
    print(calib_mdf.to_string(index=False))

    # alerts
    alert_rows: list[dict] = []
    alert_rows += _topk_alerts("prod", y_test, rank_score_test)
    alert_rows += _topk_alerts("debug_score", y_test, s_test)
    alert_rows += _topk_alerts("debug_prob", y_test, p_test_prod)
    adf = pd.DataFrame(alert_rows)

    print("\n--- Top-K Alert Summary (Test set) ---")
    print(adf.to_string(index=False))

    # coefficients
    coef = base.coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coef})
    coef_df["abs"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs", ascending=False).drop(columns=["abs"]).head(12)

    print("\nTop Base-Logit Coefficients:")
    print(coef_df.to_string(index=False))

    # save artifacts
    stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    rule_tag = (f"q{int(round(cfg.event_quantile * 100)):02d}" if cfg.event_rule == "quantile" else f"s{cfg.sigma_mult:.2f}")
    tag = f"eventlogit_{cfg.event_rule}_{rule_tag}_topk_{cfg.market}_{cfg.target_name}_h{cfg.horizon_days}d_{stamp}"
    out_dir = Path(cfg.out_dir) / tag
    _ensure_dir(out_dir)

    latest_rows.to_csv(out_dir / "rolling_calibration_drift.csv", index=False)
    rolling_calib_df.to_csv(out_dir / "rolling_calibration_drift_full.csv", index=False)

    cfg_dict = cfg.__dict__.copy()
    cfg_dict["as_of_ts"] = cfg.as_of_ts.isoformat() if cfg.as_of_ts is not None else None

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    with open(out_dir / "event_definition.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "rule": evdef.rule,
                "quantile_cut": evdef.quantile_cut,
                "vol_col": evdef.vol_col,
                "sigma_mult": cfg.sigma_mult,
                "event_quantile": cfg.event_quantile,
            },
            f,
            indent=2,
        )

    with open(out_dir / "calibration_selection.json", "w", encoding="utf-8") as f:
        json.dump(calib_scores, f, indent=2)

    with open(out_dir / "rank_selection.json", "w", encoding="utf-8") as f:
        json.dump(rank_selection, f, indent=2)

    with open(out_dir / "prior_correction.json", "w", encoding="utf-8") as f:
        json.dump(prior_correction_meta, f, indent=2)

    mdf.to_csv(out_dir / "test_metrics.csv", index=False)
    calib_mdf.to_csv(out_dir / "calib_metrics.csv", index=False)
    adf.to_csv(out_dir / "topk_alerts.csv", index=False)
    coef_df.to_csv(out_dir / "top_coefficients.csv", index=False)

    for name, rdf in rel_tables.items():
        rdf.to_csv(out_dir / f"reliability_{name}.csv", index=False)

    with open(out_dir / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "market": cfg.market,
                "target_name": cfg.target_name,
                "target_version": cfg.target_version,
                "feature_version": cfg.feature_version,
                "horizon_days": cfg.horizon_days,
                "event_rule": cfg.event_rule,
                "calib_split": cfg.calib_split,
                "platt_params": platt_params,
                "chosen_stream": chosen,
                "calib_inverted": bool(calib_inverted),
                "auc_fit_raw_for_direction": float(auc_fit_raw),
                "auc_calib_raw_diagnostic": float(auc_calib_raw) if np.isfinite(auc_calib_raw) else None,
                "as_of_ts": cfg.as_of_ts.isoformat(),
            },
            f,
            indent=2,
        )

    scored = test_df[[c for c in ["market", "as_of_date", "forward_date", "target_value"] if c in test_df.columns]].copy()
    scored["y_event"] = y_test
    scored["p_prod"] = p_test_prod
    scored["p_raw"] = p_test_raw
    scored["score_raw"] = s_test
    scored["rank_method"] = rank_method
    scored["rank_score"] = rank_score_test
    if p_test_iso is not None:
        scored["p_isotonic"] = p_test_iso
    if p_test_sig is not None:
        scored["p_sigmoid"] = p_test_sig
    scored.to_csv(out_dir / "test_scored.csv", index=False)

    print(f"\nArtifacts saved to: {out_dir.as_posix()}")



