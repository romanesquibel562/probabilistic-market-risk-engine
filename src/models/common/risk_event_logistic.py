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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression

from src.core.config import settings
from src.training.training_matrix import build_training_matrix


# ----------------------------
# Config
# ----------------------------

CalibSplit = Literal["tail", "interleaved"]
EventRule = Literal["sigma", "quantile"]
RankMethod = Literal["score", "prob"]


@dataclass
class EventLogitConfig:
    # Market is required (no SPY default)
    market: str

    # target
    horizon_days: int = 5
    target_name: str = "fwd_ret_5d_log"
    target_version: str | None = None  # defaulted from settings if None

    # features
    feature_version: str | None = None  # defaulted from settings if None

    # windows (row-count windows; you treat them as trading-days-ish)
    test_window_days: int = 90
    calib_window_days: int = 90

    # event definition
    event_rule: EventRule = "sigma"
    sigma_mult: float = 1.25
    event_quantile: float = 0.20

    # orchestrator convenience
    auto_tune_defaults: bool = True

    # calibration & production safeguards
    calib_split: CalibSplit = "interleaved"
    allow_isotonic: bool = True
    apply_prior_correction: bool = True
    prior_anchor: Literal["fit", "calib"] = "calib"  # which set to anchor base rate correction to (if enabled)

    # ranking selection (for Top-K alerts)
    choose_rank_method: bool = True
    rank_metric_k: float = 0.10  # pick ranking method by F1 at this top-k on CALIB

    fit_max_rows: int | None = None     # optional cap on FIT history to reduce regime mismatch (for long horizons)
    fit_tail_only: bool = True          # whether to fit event definition on FIT tail only (vs whole FIT)
    debug_splits: bool = False          # whether to print event rates on FIT tail vs head for debugging

    # artifact/logging
    out_dir: str = "artifacts/models"


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


def prior_correct_probs(
    p: np.ndarray,
    *,
    train_base_rate: float,
    desired_base_rate: float,
) -> np.ndarray:
    """
    Intercept-only correction in log-odds space.
    Production safeguard to prevent p_mean drifting away from anchored base rate.
    """
    eps = 1e-9
    train_base_rate = float(np.clip(train_base_rate, eps, 1 - eps))
    desired_base_rate = float(np.clip(desired_base_rate, eps, 1 - eps))

    shift = (
        np.log(desired_base_rate / (1 - desired_base_rate))
        - np.log(train_base_rate / (1 - train_base_rate))
    )
    return _sigmoid(_logit(p) + shift)


def _auto_tune_defaults(cfg: EventLogitConfig) -> EventLogitConfig:
    """
    Horizon-specific defaults.
    """
    if not cfg.auto_tune_defaults:
        return cfg

    cfg = EventLogitConfig(**cfg.__dict__)  # copy

    if cfg.horizon_days == 5:
        cfg.allow_isotonic = True
        cfg.test_window_days = 90
        cfg.calib_window_days = 252

    elif cfg.horizon_days == 21:
        cfg.allow_isotonic = False
        cfg.test_window_days = 252
        cfg.calib_window_days = 252

    elif cfg.horizon_days == 63:
        cfg.allow_isotonic = False
        cfg.apply_prior_correction = True
        cfg.test_window_days = 504
        cfg.calib_window_days = 504

    return cfg


def _p_std_min_for_horizon(h: int) -> float:
    # Longer horizons naturally have smoother probability streams.
    if h >= 63:
        return 0.01
    if h >= 21:
        return 0.02
    return 0.04


def _platt_fit(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    s = np.asarray(s, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=int)

    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(s, y)
    a = float(lr.coef_.ravel()[0])
    b = float(lr.intercept_.ravel()[0])

    # prevent ranking inversion
    if a < 0:
        a = -a
        b = -b  # sigmoid(a*s+b) == 1 - sigmoid(original)

    return a, b


def _platt_predict(s: np.ndarray, a: float, b: float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    return _sigmoid(a * s + b)


def _pick_vol_col(df: pd.DataFrame, h: int) -> str:
    """
    Pick realized vol column matching horizon if available.
    """
    candidates = [f"rv_{h}d", "rv_21d", "rv_5d"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No realized vol column found. Tried: {candidates}")


def _solve_intercept_shift(p: np.ndarray, target_rate: float) -> float:
    """
    Find shift s such that mean(sigmoid(logit(p) + s)) ~= target_rate.
    Binary search in logit space; stable and monotonic.
    """
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
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-9, 1 - 1e-9)

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
# Event definition (consistent)
# ----------------------------

@dataclass
class EventDef:
    rule: EventRule
    quantile_cut: float | None = None   # for quantile rule
    vol_col: str | None = None          # for sigma rule


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

    uniq = np.unique(y)
    has_two = len(uniq) > 1

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


def _topk_f1_at_k(y: np.ndarray, score: np.ndarray, k: float) -> tuple[float, float, float]:
    rows = _topk_alerts("tmp", y, score, ks=(k,))
    if not rows:
        return 0.0, 0.0, 0.0
    r = rows[0]
    return float(r["precision"]), float(r["recall"]), float(r["f1"])


def _select_best_calibration(
    *,
    y_calib: np.ndarray,
    p_calib_raw: np.ndarray,
    p_calib_sig: np.ndarray | None,
    p_calib_iso: np.ndarray | None,
    min_improve: float = 1e-6,
    spread_min: float = 0.05,
    p_std_min: float = 0.02,
    unique_ratio_min: float = 0.05,
    max_auc_drop: float = 0.02,
    max_ap_drop: float = 0.02,
) -> tuple[str, dict]:
    y_calib = np.asarray(y_calib, dtype=int)

    def clip(p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)

    def score_calib(p: np.ndarray) -> dict:
        p = clip(p)
        b = float(brier_score_loss(y_calib, p))
        ll = float(log_loss(y_calib, p, labels=[0, 1]))

        q05, q95 = np.quantile(p, [0.05, 0.95])
        spread = float(q95 - q05)

        if _has_both_classes(y_calib):
            auc = float(roc_auc_score(y_calib, p))
            ap = float(average_precision_score(y_calib, p))
        else:
            auc = float("nan")
            ap = float("nan")

        p_std = float(np.std(p))
        unique_ratio = float(len(np.unique(np.round(p, 6))) / max(1, len(p)))

        return {
            "brier": b,
            "log_loss": ll,
            "spread_p95_p05": spread,
            "p_std": p_std,
            "unique_ratio": unique_ratio,
            "auc": auc,
            "ap": ap,
        }

    scores: dict[str, dict[str, float]] = {}

    scores["raw"] = score_calib(p_calib_raw)
    raw = scores["raw"]

    candidates: list[str] = ["raw"]
    if p_calib_sig is not None:
        scores["sigmoid"] = score_calib(p_calib_sig)
        candidates.append("sigmoid")
    if p_calib_iso is not None:
        scores["isotonic"] = score_calib(p_calib_iso)
        candidates.append("isotonic")

    rejected: dict[str, str] = {}
    viable: list[str] = ["raw"]

    for name in candidates:
        if name == "raw":
            continue
        s = scores[name]

        if s["spread_p95_p05"] < spread_min:
            rejected[name] = f"spread<{spread_min:.3f}"
            continue
        if s.get("p_std", 1.0) < p_std_min:
            rejected[name] = f"p_std<{p_std_min:.3f}"
            continue
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

    best_name = sorted(viable, key=lambda n: (scores[n]["brier"], scores[n]["log_loss"]))[0]

    if best_name != "raw":
        if (raw["brier"] - scores[best_name]["brier"]) < float(min_improve):
            best_name = "raw"

    scores["rejected"] = rejected
    scores["chosen"] = {"name": best_name}
    return best_name, scores


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


# ----------------------------
# Main runner
# ----------------------------

def run_step6_event_logistic(cfg: EventLogitConfig) -> None:
    cfg = _auto_tune_defaults(cfg)

    # Fill versions from settings if not explicitly provided
    if cfg.feature_version is None:
        cfg.feature_version = settings.DEFAULT_FEATURE_VERSION
    if cfg.target_version is None:
        cfg.target_version = settings.DEFAULT_TARGET_VERSION

    _validate_target_pair(cfg.target_name, cfg.horizon_days)

    prior_correction_meta = {"enabled": False}

    print("\n=== STEP 6.1: Logistic Risk-Event + Calibration Guard + Top-K ===")
    print(
        "Config: "
        f"market={cfg.market} target={cfg.target_name} h={cfg.horizon_days} "
        f"rule={cfg.event_rule} sigma_mult={cfg.sigma_mult} "
        f"test={cfg.test_window_days} calib={cfg.calib_window_days} "
        f"calib_split={cfg.calib_split} allow_iso={cfg.allow_isotonic}"
    )

    # 1) build matrix
    df = build_training_matrix(
        market=cfg.market,
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
        fit_max_rows=getattr(cfg, "fit_max_rows", None),
        fit_tail_only=getattr(cfg, "fit_tail_only", True),
        debug_splits=getattr(cfg, "debug_splits", False),
    )

    print(f"Rows total: {len(df)} | Fit: {len(fit_df)} | Calib: {len(calib_df)} | Test: {len(test_df)}")

    # 3) fit event definition on FIT only
    evdef = _fit_event_definition(fit_df, cfg=cfg)

    # 4) labels (consistent)
    y_fit, thresh_fit = _make_event_labels(fit_df, cfg=cfg, evdef=evdef)
    y_calib, thresh_calib = _make_event_labels(calib_df, cfg=cfg, evdef=evdef)
    y_test, thresh_test = _make_event_labels(test_df, cfg=cfg, evdef=evdef)

    # pretty print event rule
    if cfg.event_rule == "sigma":
        assert evdef.vol_col is not None
        print(
            f"Event rule: y_fwd <= -{cfg.sigma_mult:.3f} * vol_asof(h={cfg.horizon_days}d)  [vol_col={evdef.vol_col}]"
        )
    else:
        assert evdef.quantile_cut is not None
        print(f"Event rule: y_fwd <= quantile_cut(q={cfg.event_quantile:.2f}) = {evdef.quantile_cut:.6f} (fit-only)")

    print(f"Event rates: fit={y_fit.mean():.3f}, calib={y_calib.mean():.3f}, test={y_test.mean():.3f}")

    if cfg.event_rule == "sigma":
        ts = pd.Series(thresh_test).dropna()
        if not ts.empty:
            q05, q50, q95 = np.quantile(ts.values, [0.05, 0.50, 0.95])
            print(f"Threshold stats (test): mean={ts.mean():.6f} p05={q05:.6f} p50={q50:.6f} p95={q95:.6f}")
    else:
        ts = pd.Series(thresh_test).dropna()
        if not ts.empty:
            print(f"Threshold (quantile cut): {float(ts.iloc[0]):.6f}")

    # 5) model
    feature_cols = [c for c in df.columns if c not in ("market", "as_of_date", "target_value")]
    X_fit = fit_df[feature_cols].astype(float).values
    X_calib = calib_df[feature_cols].astype(float).values
    X_test = test_df[feature_cols].astype(float).values

    cw = "balanced" if cfg.horizon_days == 5 else None

    base = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight=cw,
    )
    base.fit(X_fit, y_fit)

    s_fit = base.decision_function(X_fit)
    s_calib = base.decision_function(X_calib)
    s_test = base.decision_function(X_test)

    p_fit_raw = base.predict_proba(X_fit)[:, 1]
    p_calib_raw = base.predict_proba(X_calib)[:, 1]
    p_test_raw = base.predict_proba(X_test)[:, 1]

    print(f"p_calib_raw: {_prob_stats(p_calib_raw)}")
    print(f"p_test_raw:  {_prob_stats(p_test_raw)}")
    print(f"Class rates: y_fit={y_fit.mean():.3f} | y_calib={y_calib.mean():.3f} | y_test={y_test.mean():.3f}")

    can_calibrate = _has_both_classes(y_calib)

    p_calib_iso: np.ndarray | None = None
    p_test_iso: np.ndarray | None = None
    p_calib_sig: np.ndarray | None = None
    p_test_sig: np.ndarray | None = None
    a: float | None = None
    b: float | None = None
    iso: IsotonicRegression | None = None

    if not can_calibrate:
        print("WARNING: calib split has a single class; skipping isotonic/sigmoid calibration.")
    else:
        if cfg.allow_isotonic:
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_calib_raw, y_calib)
                p_calib_iso = iso.predict(p_calib_raw)
                p_test_iso = iso.predict(p_test_raw)
            except Exception as e:
                print(f"Calibration isotonic skipped due to error: {e}")
                p_calib_iso = None
                p_test_iso = None
                iso = None
        else:
            print("Isotonic disabled by config (recommended for h>=21).")

        try:
            a, b = _platt_fit(s_calib, y_calib)
            print(f"Platt params: a={a:.6f} b={b:.6f}")
            p_calib_sig = _platt_predict(s_calib, a, b)
            p_test_sig = _platt_predict(s_test, a, b)
        except Exception as e:
            print(f"Calibration sigmoid skipped due to error: {e}")
            p_calib_sig = None
            p_test_sig = None

    chosen = "raw"
    calib_scores = {"chosen": {"name": "raw"}}
    if can_calibrate:
        chosen, calib_scores = _select_best_calibration(
            y_calib=y_calib,
            p_calib_raw=p_calib_raw,
            p_calib_sig=p_calib_sig,
            p_calib_iso=p_calib_iso,
            min_improve=1e-6,
            p_std_min=_p_std_min_for_horizon(cfg.horizon_days),
            unique_ratio_min=0.10 if cfg.horizon_days == 5 else 0.05,
        )

    p_calib_prod = _build_prod_probs_from_choice(chosen=chosen, p_raw=p_calib_raw, p_sig=p_calib_sig, p_iso=p_calib_iso)
    p_test_prod = _build_prod_probs_from_choice(chosen=chosen, p_raw=p_test_raw, p_sig=p_test_sig, p_iso=p_test_iso)

    print(f"Calibration selection (by calib set): chosen={chosen} | scores={calib_scores}")

    p_fit_sig: np.ndarray | None = None
    p_fit_iso: np.ndarray | None = None
    if a is not None and b is not None:
        p_fit_sig = _platt_predict(s_fit, a, b)
    if iso is not None:
        p_fit_iso = iso.predict(p_fit_raw)

    p_fit_prod = _build_prod_probs_from_choice(chosen=chosen, p_raw=p_fit_raw, p_sig=p_fit_sig, p_iso=p_fit_iso)

    platt_params = {"a": float(a), "b": float(b)} if (can_calibrate and a is not None and b is not None) else None

    # 7) base-rate alignment
    if cfg.apply_prior_correction:
        anchor = cfg.prior_anchor  # "fit" or "calib"
        if anchor == "fit":
            y_anchor = y_fit
            p_anchor = p_fit_prod
        else:
            y_anchor = y_calib
            p_anchor = p_calib_prod

        pi_anchor = float(np.mean(y_anchor))
        pbar_anchor = float(np.mean(p_anchor))
        shift = _solve_intercept_shift(p_anchor, pi_anchor)

        prior_correction_meta = {
            "enabled": True,
            "anchor": anchor,
            "horizon_days": int(cfg.horizon_days),
            "chosen_stream": str(chosen),
            "pi_anchor": float(pi_anchor),
            "pbar_anchor": float(pbar_anchor),
            "shift_logit": float(shift),
            "calib_mean_before": float(np.mean(p_calib_prod)),
            "test_mean_before": float(np.mean(p_test_prod)),
        }

        def _brier_ll(y, p):
            p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
            return float(brier_score_loss(y, p)), float(log_loss(y, p, labels=[0, 1]))

        b_before, ll_before = _brier_ll(y_anchor, p_anchor)
        p_anchor_after = _apply_intercept_shift(p_anchor, shift)
        b_after, ll_after = _brier_ll(y_anchor, p_anchor_after)

        apply_shift = (b_after <= b_before + 1e-6) and (ll_after <= ll_before + 1e-6)

        prior_correction_meta.update(
            {
                "anchor_brier_before": b_before,
                "anchor_logloss_before": ll_before,
                "anchor_brier_after": b_after,
                "anchor_logloss_after": ll_after,
                "applied": bool(apply_shift),
            }
        )

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
    else:
        prior_correction_meta = {"enabled": False}

    # 8B) reliability tables
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

    if not _has_both_classes(y_test):
        print("NOTE: test set has a single class; reliability table may be less informative.")
    if not _has_both_classes(y_calib):
        print("NOTE: calib set has a single class; reliability table may be less informative.")

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

        rolling_frames.append(
            _rolling_calibration(test_df, y_test, p_test_prod, window=w, min_periods=mp, label=f"test_prod_w{w}")
        )
        rolling_frames.append(
            _rolling_calibration(test_df, y_test, p_test_raw, window=w, min_periods=mp, label=f"test_raw_w{w}")
        )

    rolling_calib_df = pd.concat(rolling_frames, ignore_index=True)

    print("\n--- Rolling Calibration Drift (latest per stream) ---")
    latest_rows = (
        rolling_calib_df.sort_values("as_of_date").groupby("label", as_index=False).tail(1).sort_values("label")
    )
    print(latest_rows.to_string(index=False))

    # ranking (Option A): ALWAYS rank by raw score for alerts
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

    # coefficients (top abs)
    coef = base.coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coef})
    coef_df["abs"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs", ascending=False).drop(columns=["abs"]).head(12)

    print("\nTop Base-Logit Coefficients:")
    print(coef_df.to_string(index=False))

    # save artifacts
    stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    tag = f"eventlogit_{cfg.event_rule}_topk_{cfg.market}_{cfg.target_name}_h{cfg.horizon_days}d_{stamp}"
    out_dir = Path(cfg.out_dir) / tag
    _ensure_dir(out_dir)
    
    latest_rows.to_csv(out_dir / "rolling_calibration_drift.csv", index=False)
    rolling_calib_df.to_csv(out_dir / "rolling_calibration_drift_full.csv", index=False)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)

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
    adf.to_csv(out_dir / "topk_alerts.csv", index=False)
    coef_df.to_csv(out_dir / "top_coefficients.csv", index=False)

    for name, rdf in rel_tables.items():
        rdf.to_csv(out_dir / f"reliability_{name}.csv", index=False)

    rolling_calib_df.to_csv(out_dir / "rolling_calibration_test.csv", index=False)
    calib_mdf.to_csv(out_dir / "calib_metrics.csv", index=False)

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
                "calib_best_by_brier": calib_mdf.iloc[0].to_dict() if not calib_mdf.empty else None,
                "test_best_by_brier": mdf.iloc[0].to_dict() if not mdf.empty else None,
                "calibration_selection": calib_scores,
                "rank_method": "score",
                "rank_reason": "always_rank_by_score_option_A",
            },
            f,
            indent=2,
        )

    scored = test_df[["market", "as_of_date", "target_value"]].copy()
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





