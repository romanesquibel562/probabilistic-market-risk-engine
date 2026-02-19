# src/models/backtest.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.core.config import settings
from src.training.training_matrix import build_training_matrix
from src.models.common.risk_event_logistic import (
    EventLogitConfig,
    _auto_tune_defaults,
    _fit_event_definition,
    _make_event_labels,
    _split_fit_calib_test,
    _platt_fit,
    _platt_predict,
    _select_best_calibration,
    _build_prod_probs_from_choice,
    _solve_intercept_shift,
    _apply_intercept_shift,
    _has_both_classes,
    _p_std_min_for_horizon,
    _reliability_table,
)

# ----------------------------
# Backtest config
# ----------------------------

@dataclass
class WalkForwardConfig:
    market: str = "SPY"
    horizons: tuple[int, ...] = (5, 21, 63)

    # Fold movement
    step_size: int = 21  # advance anchor by ~1 month each fold

    # Where to start folds (need enough history for fit+calib+test)
    min_train_rows: int = 750  # minimum fit history before first fold

    # Save
    out_dir: str = "artifacts/backtests"
    tag: str = "wf_eventlogit_v1"

    # Diagnostics
    save_reliability: bool = True
    reliability_bins: int = 10


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)


def _metrics_safe(tag: str, y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = _clip_probs(p)

    out = {
        "tag": tag,
        "n": int(len(y)),
        "event_rate": float(y.mean()) if len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else float("nan"),
        "p_mean": float(p.mean()) if len(y) else float("nan"),
        "p_std": float(np.std(p)) if len(y) else float("nan"),
    }

    if _has_both_classes(y):
        out["auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    else:
        out["auc"] = float("nan")
        out["ap"] = float("nan")

    return out


def _fit_score_fold(
    df_fold: pd.DataFrame,
    *,
    cfg: EventLogitConfig,
    fold_id: int,
    save_reliability: bool,
    reliability_bins: int,
) -> tuple[dict, dict[str, pd.DataFrame] | None]:
    """
    Run your Step-6 logic on ONE fold slice df_fold (already chronological).

    Returns:
      - fold_metrics dict
      - reliability tables dict (optional)
    """
    cfg = _auto_tune_defaults(cfg)

    # 1) split within the fold slice
    fit_df, calib_df, test_df = _split_fit_calib_test(
        df_fold,
        test_n=int(cfg.test_window_days),
        calib_n=int(cfg.calib_window_days),
        mode=cfg.calib_split,
        fit_max_rows=getattr(cfg, "fit_max_rows", None),
        fit_tail_only=getattr(cfg, "fit_tail_only", True),
        debug_splits=False,
    )

    # 2) event definition + labels
    evdef = _fit_event_definition(fit_df, cfg=cfg)
    y_fit, _ = _make_event_labels(fit_df, cfg=cfg, evdef=evdef)
    y_calib, _ = _make_event_labels(calib_df, cfg=cfg, evdef=evdef)
    y_test, _ = _make_event_labels(test_df, cfg=cfg, evdef=evdef)

    feature_cols = [c for c in df_fold.columns if c not in ("market", "as_of_date", "target_value")]
    X_fit = fit_df[feature_cols].astype(float).values
    X_calib = calib_df[feature_cols].astype(float).values
    X_test = test_df[feature_cols].astype(float).values

    base = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
    )
    base.fit(X_fit, y_fit)

    s_fit = base.decision_function(X_fit)
    s_calib = base.decision_function(X_calib)
    s_test = base.decision_function(X_test)

    p_fit_raw = base.predict_proba(X_fit)[:, 1]
    p_calib_raw = base.predict_proba(X_calib)[:, 1]
    p_test_raw = base.predict_proba(X_test)[:, 1]

    # 3) calibration candidates
    can_calibrate = _has_both_classes(y_calib)

    p_calib_iso = p_test_iso = None
    p_calib_sig = p_test_sig = None
    a = b = None
    iso = None

    if can_calibrate:
        if cfg.allow_isotonic:
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_calib_raw, y_calib)
                p_calib_iso = iso.predict(p_calib_raw)
                p_test_iso = iso.predict(p_test_raw)
            except Exception:
                iso = None
                p_calib_iso = None
                p_test_iso = None

        try:
            a, b = _platt_fit(s_calib, y_calib)
            p_calib_sig = _platt_predict(s_calib, a, b)
            p_test_sig = _platt_predict(s_test, a, b)
        except Exception:
            a = b = None
            p_calib_sig = None
            p_test_sig = None

    # 4) choose stream on calib
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

    p_calib_prod = _build_prod_probs_from_choice(
        chosen=chosen, p_raw=p_calib_raw, p_sig=p_calib_sig, p_iso=p_calib_iso
    )
    p_test_prod = _build_prod_probs_from_choice(
        chosen=chosen, p_raw=p_test_raw, p_sig=p_test_sig, p_iso=p_test_iso
    )

    # Build FIT prod probs to support prior anchor="fit"
    p_fit_sig = None
    p_fit_iso = None
    if a is not None and b is not None:
        p_fit_sig = _platt_predict(s_fit, a, b)
    if iso is not None:
        p_fit_iso = iso.predict(p_fit_raw)
    p_fit_prod = _build_prod_probs_from_choice(
        chosen=chosen, p_raw=p_fit_raw, p_sig=p_fit_sig, p_iso=p_fit_iso
    )

    # 5) prior alignment (simplified do-no-harm gate)
    prior_meta: dict = {"enabled": False}
    apply_shift = False
    shift = 0.0

    if cfg.apply_prior_correction:
        anchor = cfg.prior_anchor
        if anchor == "fit":
            y_anchor = y_fit
            p_anchor = p_fit_prod
        else:
            y_anchor = y_calib
            p_anchor = p_calib_prod

        pi_anchor = float(np.mean(y_anchor))
        pbar_anchor = float(np.mean(p_anchor))
        shift = float(_solve_intercept_shift(p_anchor, pi_anchor))

        b_before = float(brier_score_loss(y_anchor, _clip_probs(p_anchor)))
        ll_before = float(log_loss(y_anchor, _clip_probs(p_anchor), labels=[0, 1]))
        p_anchor_after = _apply_intercept_shift(p_anchor, shift)
        b_after = float(brier_score_loss(y_anchor, _clip_probs(p_anchor_after)))
        ll_after = float(log_loss(y_anchor, _clip_probs(p_anchor_after), labels=[0, 1]))

        apply_shift = (b_after <= b_before + 1e-6) and (ll_after <= ll_before + 1e-6)

        prior_meta = {
            "enabled": True,
            "anchor": anchor,
            "pi_anchor": pi_anchor,
            "pbar_anchor": pbar_anchor,
            "shift_logit": shift,
            "anchor_brier_before": b_before,
            "anchor_logloss_before": ll_before,
            "anchor_brier_after": b_after,
            "anchor_logloss_after": ll_after,
            "applied": bool(apply_shift),
        }

        if apply_shift:
            p_fit_prod = _apply_intercept_shift(p_fit_prod, shift)
            p_calib_prod = _apply_intercept_shift(p_calib_prod, shift)
            p_test_prod = _apply_intercept_shift(p_test_prod, shift)

        prior_meta["calib_mean_after"] = float(np.mean(p_calib_prod))
        prior_meta["test_mean_after"] = float(np.mean(p_test_prod))

    # 6) metrics
    m_test_prod = _metrics_safe("test_prod", y_test, p_test_prod)
    m_test_raw = _metrics_safe("test_raw", y_test, p_test_raw)
    m_calib_prod = _metrics_safe("calib_prod", y_calib, p_calib_prod)
    m_calib_raw = _metrics_safe("calib_raw", y_calib, p_calib_raw)

    fold_meta = {
        "fold_id": int(fold_id),
        "horizon_days": int(cfg.horizon_days),
        "as_of_start": str(pd.to_datetime(df_fold["as_of_date"].min()).date()),
        "as_of_end": str(pd.to_datetime(df_fold["as_of_date"].max()).date()),
        "rows_fold": int(len(df_fold)),
        "rows_fit": int(len(fit_df)),
        "rows_calib": int(len(calib_df)),
        "rows_test": int(len(test_df)),
        "event_rate_fit": float(y_fit.mean()),
        "event_rate_calib": float(y_calib.mean()),
        "event_rate_test": float(y_test.mean()),
        "chosen_stream": str(chosen),
        "calib_scores": calib_scores,
        "prior_align": prior_meta,
        **{f"test_prod_{k}": v for k, v in m_test_prod.items() if k != "tag"},
        **{f"test_raw_{k}": v for k, v in m_test_raw.items() if k != "tag"},
        **{f"calib_prod_{k}": v for k, v in m_calib_prod.items() if k != "tag"},
        **{f"calib_raw_{k}": v for k, v in m_calib_raw.items() if k != "tag"},
    }

    rel_tables = None
    if save_reliability:
        rel_tables = {
            "test_prod": _reliability_table(y_test, p_test_prod, n_bins=reliability_bins),
            "test_raw": _reliability_table(y_test, p_test_raw, n_bins=reliability_bins),
            "calib_prod": _reliability_table(y_calib, p_calib_prod, n_bins=reliability_bins),
            "calib_raw": _reliability_table(y_calib, p_calib_raw, n_bins=reliability_bins),
        }

    return fold_meta, rel_tables


def run_walk_forward_event_backtest(
    *,
    wf: WalkForwardConfig,
    base_cfg_by_horizon: dict[int, EventLogitConfig] | None = None,
) -> Path:
    """
    Main entry: walk-forward backtest across horizons.
    Returns output directory path.
    """
    stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(wf.out_dir) / f"{wf.tag}_{wf.market}_{stamp}"
    _ensure_dir(out_dir)

    with open(out_dir / "walkforward_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(wf), f, indent=2)

    all_fold_rows: list[dict] = []
    rel_rows: list[pd.DataFrame] = []

    for h in wf.horizons:
        h = int(h)

        # Build a deterministic config per horizon (don’t mutate a shared cfg)
        base_cfg = (base_cfg_by_horizon or {}).get(h)
        if base_cfg is None:
            base_cfg = EventLogitConfig(
                market=wf.market,
                horizon_days=h,
                target_name=f"fwd_ret_{h}d_log",
                feature_version=settings.DEFAULT_FEATURE_VERSION,
                target_version=settings.DEFAULT_TARGET_VERSION,
            )
        else:
            # ensure versions are set even if caller forgot
            if getattr(base_cfg, "feature_version", None) is None:
                base_cfg.feature_version = settings.DEFAULT_FEATURE_VERSION
            if getattr(base_cfg, "target_version", None) is None:
                base_cfg.target_version = settings.DEFAULT_TARGET_VERSION
            base_cfg.horizon_days = h
            base_cfg.target_name = f"fwd_ret_{h}d_log"

        cfg = _auto_tune_defaults(base_cfg)

        # Build full matrix once per horizon
        df = build_training_matrix(
            market=cfg.market,
            feature_version=cfg.feature_version,
            target_name=cfg.target_name,
            horizon_days=cfg.horizon_days,
            target_version=cfg.target_version,
            dropna_features=True,
        )
        if df.empty:
            raise ValueError(f"Empty training matrix for horizon={h}")

        df = df.sort_values("as_of_date").reset_index(drop=True)

        need = int(wf.min_train_rows + cfg.calib_window_days + cfg.test_window_days + 50)
        if len(df) < need:
            raise ValueError(f"Not enough rows for walk-forward. rows={len(df)} need >= {need} (h={h})")

        fold_id = 0
        for end in range(need, len(df) + 1, int(wf.step_size)):
            df_fold = df.iloc[:end].copy()

            fold_meta, rel = _fit_score_fold(
                df_fold,
                cfg=cfg,
                fold_id=fold_id,
                save_reliability=wf.save_reliability,
                reliability_bins=wf.reliability_bins,
            )
            all_fold_rows.append(fold_meta)

            if rel is not None:
                for name, rdf in rel.items():
                    rr = rdf.copy()
                    rr["fold_id"] = fold_id
                    rr["horizon_days"] = h
                    rr["stream"] = name
                    rel_rows.append(rr)

            fold_id += 1

        horizon_df = pd.DataFrame([r for r in all_fold_rows if r["horizon_days"] == h])
        horizon_df.to_csv(out_dir / f"fold_metrics_h{h}.csv", index=False)

    fold_df = pd.DataFrame(all_fold_rows)
    fold_df.to_csv(out_dir / "fold_metrics_all.csv", index=False)

    if rel_rows:
        rel_df = pd.concat(rel_rows, ignore_index=True)
        rel_df.to_csv(out_dir / "reliability_all_folds.csv", index=False)

    # Summary
    summary: dict[int, dict] = {}
    for h in wf.horizons:
        h = int(h)
        sub = fold_df[fold_df["horizon_days"] == h]
        if sub.empty:
            continue

        # prior_align is a column of dicts
        applied_vals = []
        for v in sub.get("prior_align", []):
            if isinstance(v, dict):
                applied_vals.append(bool(v.get("applied", False)))
        prior_applied_rate = float(np.mean(applied_vals)) if applied_vals else float("nan")

        summary[h] = {
            "n_folds": int(len(sub)),
            "mean_test_brier_prod": float(sub["test_prod_brier"].mean()),
            "mean_test_logloss_prod": float(sub["test_prod_log_loss"].mean()),
            "mean_test_auc_prod": float(sub["test_prod_auc"].mean(skipna=True)),
            "mean_test_ap_prod": float(sub["test_prod_ap"].mean(skipna=True)),
            "mean_test_pmean_prod": float(sub["test_prod_p_mean"].mean()),
            "mean_test_event_rate": float(sub["event_rate_test"].mean()),
            "chosen_stream_counts": sub["chosen_stream"].value_counts().to_dict(),
            "prior_applied_rate": prior_applied_rate,
        }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[walkforward] saved -> {out_dir.as_posix()}")
    return out_dir


if __name__ == "__main__":
    wf = WalkForwardConfig(
        market="SPY",
        horizons=(5, 21, 63),
        step_size=21,
        min_train_rows=750,
        tag="wf_eventlogit_v1",
        save_reliability=True,
        reliability_bins=10,
    )

    run_walk_forward_event_backtest(wf=wf)

# Run:
#   python -m src.models.backtest
