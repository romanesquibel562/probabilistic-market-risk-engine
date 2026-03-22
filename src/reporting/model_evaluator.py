# src/reporting/model_evaluator.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class EvalScanConfig:
    repo_root: Path
    artifacts_models_dir: Path
    artifacts_backtests_dir: Path
    artifacts_outputs_dir: Path
    latest_only: bool = True  # keep only latest run per (market,target,horizon,rule)


# -----------------------------
# Helpers
# -----------------------------
def _safe_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def _parse_from_dirname(dirname: str) -> dict:
    """
    Best-effort metadata parsing from artifact directory naming.

    Examples:
      eventlogit_calib_topk_SPY_fwd_ret_5d_log_20260212_034415
      eventlogit_sigma_s1.25_topk_SPY_fwd_ret_63d_log_h63d_20260301_050541
      eventlogit_quantile_q10_topk_SPY_fwd_ret_21d_log_h21d_20260301_050533
    """
    meta = {
        "market": None,
        "target_name": None,
        "horizon_days": None,
        "event_rule": None,
        "run_date": None,
        "run_time": None,
        "run_id": dirname,
    }

    # market token (SPY, QQQ, etc.)
    m = re.search(r"_([A-Z]{2,6})_", dirname)
    if m:
        meta["market"] = m.group(1)

    # event rule present in newer names
    r = re.search(r"_(sigma|quantile)_", dirname)
    if r:
        meta["event_rule"] = r.group(1)

    # target name + run date
    # Handles:
    #   ..._fwd_ret_5d_log_YYYYMMDD_...
    #   ..._fwd_ret_5d_log_h5d_YYYYMMDD_...
    t = re.search(r"_(fwd_[a-zA-Z0-9_]+?)_(\d{8})_", dirname)
    if t:
        meta["target_name"] = t.group(1)
        meta["run_date"] = t.group(2)

    # If target name includes _h5d suffix (from your newer naming), normalize it away:
    #   fwd_ret_5d_log_h5d -> fwd_ret_5d_log
    if meta["target_name"] and re.search(r"_h\d+d$", meta["target_name"]):
        meta["target_name"] = re.sub(r"_h\d+d$", "", meta["target_name"])

    # Infer horizon from target_name if possible
    if meta["target_name"]:
        hh = re.search(r"fwd_ret_(\d+)d_", meta["target_name"])
        if hh:
            meta["horizon_days"] = int(hh.group(1))

    # fallback: capture run_date/run_time at end
    t2 = re.search(r"_(\d{8})_(\d{6})$", dirname)
    if t2:
        meta["run_date"] = meta["run_date"] or t2.group(1)
        meta["run_time"] = t2.group(2)

    # Legacy naming: eventlogit_calib_topk_* has no rule in folder name
    # Tag them so grouping works instead of leaving blank.
    if meta["event_rule"] is None and dirname.startswith("eventlogit_calib_topk_"):
        meta["event_rule"] = "legacy"

    return meta


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _read_first_existing_csv(run_dir: Path, candidates: list[str]) -> Optional[pd.DataFrame]:
    for name in candidates:
        df = _read_csv_if_exists(run_dir / name)
        if df is not None and not df.empty:
            return df
    return None


def _read_by_glob(run_dir: Path, patterns: list[str]) -> Optional[pd.DataFrame]:
    """
    Try reading the first CSV that matches any glob pattern in the run dir.
    """
    for pat in patterns:
        matches = sorted(run_dir.glob(pat))
        for p in matches:
            if p.is_file() and p.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(p)
                    if df is not None and not df.empty:
                        return df
                except Exception:
                    continue
    return None


def _pick_metrics_row(df: pd.DataFrame) -> pd.Series:
    """
    Prefer: test_prod -> test_raw -> calib_prod -> calib_raw -> first row
    """
    if df is None or df.empty:
        raise ValueError("metrics dataframe is empty")

    if "tag" not in df.columns:
        return df.iloc[0]

    preferred = ["test_prod", "test_raw", "calib_prod", "calib_raw"]
    for t in preferred:
        sub = df[df["tag"] == t]
        if not sub.empty:
            return sub.iloc[0]

    return df.iloc[0]


def _summarize_drift(drift_df: Optional[pd.DataFrame]) -> dict:
    out: dict = {}
    if drift_df is None or drift_df.empty:
        return out

    out["drift_rows"] = int(len(drift_df))

    # Your drift files currently have roll_event_rate and roll_p_mean; derive a gap if needed.
    if "roll_event_rate" in drift_df.columns and "roll_p_mean" in drift_df.columns:
        s = pd.to_numeric(drift_df["roll_event_rate"], errors="coerce") - pd.to_numeric(
            drift_df["roll_p_mean"], errors="coerce"
        )
        s = s.dropna()
        if not s.empty:
            out["drift_gap_last"] = float(s.iloc[-1])
            out["drift_gap_mean"] = float(s.mean())
            out["drift_gap_abs_mean"] = float(s.abs().mean())
            out["drift_gap_abs_max"] = float(s.abs().max())
        return out

    # Fallback: try to find an explicit gap column if present
    possible_cols = [
        "calibration_gap",
        "gap",
        "roll_gap",
        "rolling_gap",
        "gap_roll",
        "p_mean_minus_event_rate",
        "event_rate_minus_p_mean",
    ]
    gap_col = next((c for c in possible_cols if c in drift_df.columns), None)
    if gap_col is None:
        return out

    s = pd.to_numeric(drift_df[gap_col], errors="coerce").dropna()
    if s.empty:
        return out

    out["drift_gap_last"] = float(s.iloc[-1])
    out["drift_gap_mean"] = float(s.mean())
    out["drift_gap_abs_mean"] = float(s.abs().mean())
    out["drift_gap_abs_max"] = float(s.abs().max())
    out["drift_gap_col"] = gap_col
    return out


def _summarize_topk(topk_df: Optional[pd.DataFrame]) -> dict:
    out: dict = {}
    if topk_df is None or topk_df.empty:
        return out

    out["topk_rows"] = int(len(topk_df))

    # You save tables like:
    # tag, k, alert_rate, precision, recall, f1, tp, fp, tn, fn
    cols = [
        "k",
        "alert_rate",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
        "tag",
    ]
    last = topk_df.iloc[-1].to_dict()

    for c in cols:
        if c in topk_df.columns:
            val = last.get(c)
            key = f"topk_{c}_last"
            try:
                out[key] = float(val)
            except Exception:
                out[key] = str(val)

    # Keep which row tag we used (e.g., prod_top10%)
    if "tag" in topk_df.columns:
        out["topk_row_tag_used"] = str(last.get("tag"))

    return out


def _merge_run_meta(row: dict, run_meta_df: Optional[pd.DataFrame]) -> dict:
    if run_meta_df is None or run_meta_df.empty:
        return row

    meta = run_meta_df.iloc[0].to_dict()
    for k in [
        "market",
        "target_name",
        "horizon_days",
        "event_rule",
        "feature_version",
        "target_version",
        "event_family",
        "event_params",
        "as_of_date",
        "series_id",
    ]:
        if k in meta and pd.notna(meta[k]):
            row[k] = meta[k]

    # normalize target_name if it came in as fwd_ret_5d_log_h5d
    tn = row.get("target_name")
    if isinstance(tn, str) and re.search(r"_h\d+d$", tn):
        row["target_name"] = re.sub(r"_h\d+d$", "", tn)

    # infer horizon if missing
    if row.get("horizon_days") in (None, "", -1) and isinstance(row.get("target_name"), str):
        hh = re.search(r"fwd_ret_(\d+)d_", row["target_name"])
        if hh:
            row["horizon_days"] = int(hh.group(1))

    return row


# -----------------------------
# Phase 7: scan model artifacts
# -----------------------------
def build_model_eval_summary(
    market: Optional[str] = None,
    repo_root: Optional[Path] = None,
    latest_only: bool = True,
) -> pd.DataFrame:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    models_dir = repo_root / "artifacts" / "models"

    run_dirs = sorted([p for p in models_dir.glob("eventlogit*") if p.is_dir()])

    rows: list[dict] = []
    for run_dir in run_dirs:
        meta = _parse_from_dirname(run_dir.name)
        if market and meta.get("market") and meta["market"] != market:
            continue

        row = {**meta, "artifact_dir": str(run_dir)}

        # run_meta (optional)
        run_meta_df = _read_first_existing_csv(run_dir, candidates=["run_meta.csv"])
        row = _merge_run_meta(row, run_meta_df)

        # METRICS
        metrics_df = _read_first_existing_csv(
            run_dir,
            candidates=[
                "metrics.csv",
                "test_metrics.csv",
                "metrics_test.csv",
                "probability_metrics.csv",
            ],
        )
        if metrics_df is None:
            metrics_df = _read_by_glob(run_dir, patterns=["*metrics*.csv"])

        # DRIFT
        drift_df = _read_first_existing_csv(
            run_dir,
            candidates=[
                "rolling_calibration_drift.csv",
                "rolling_calibration_drift_full.csv",
                "calibration_drift.csv",
            ],
        )
        if drift_df is None:
            drift_df = _read_by_glob(run_dir, patterns=["*drift*.csv"])

        # TOPK
        topk_df = _read_first_existing_csv(
            run_dir,
            candidates=["thresholds_topk.csv", "topk_alerts.csv", "alerts_topk.csv"],
        )
        if topk_df is None:
            topk_df = _read_by_glob(run_dir, patterns=["*topk*.csv", "*threshold*.csv"])

        # Fill metrics fields
        if metrics_df is not None and not metrics_df.empty:
            picked = _pick_metrics_row(metrics_df)
            row["picked_tag"] = str(picked.get("tag", "unknown"))
            for col in metrics_df.columns:
                row[f"m_{col}"] = picked.get(col)

        # Optional summaries
        row.update(_summarize_drift(drift_df))
        row.update(_summarize_topk(topk_df))

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Ensure consistent keys for grouping
    for c in ["market", "target_name", "horizon_days", "event_rule"]:
        if c not in out.columns:
            out[c] = None

    # Normalize target_name suffix
    out["target_name"] = out["target_name"].astype(str).str.replace(r"_h\d+d$", "", regex=True)

    # Normalize horizon
    if "horizon_days" in out.columns:
        out["horizon_days"] = pd.to_numeric(out["horizon_days"], errors="coerce").fillna(-1).astype(int)

    if latest_only:
        run_date = out.get("run_date", pd.Series([""] * len(out))).fillna("")
        run_time = out.get("run_time", pd.Series([""] * len(out))).fillna("")
        out["_sortkey"] = run_date + "_" + run_time

        out["_g_market"] = out["market"].fillna("UNK")
        out["_g_target"] = out["target_name"].fillna("UNK")
        out["_g_h"] = out["horizon_days"].fillna(-1)
        out["_g_rule"] = out["event_rule"].fillna("UNK")

        out = (
            out.sort_values(["_sortkey"], ascending=True)
            .groupby(["_g_market", "_g_target", "_g_h", "_g_rule"], as_index=False)
            .tail(1)
            .drop(columns=[c for c in out.columns if c.startswith("_g_") or c == "_sortkey"])
        )

    return out


def write_eval_outputs(
    df: pd.DataFrame,
    market: str,
    as_of_date: str,
    repo_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    outputs_dir = repo_root / "artifacts" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = outputs_dir / f"model_eval_summary_{market}_{as_of_date}.csv"
    md_path = outputs_dir / f"model_eval_report_{market}_{as_of_date}.md"

    df.to_csv(csv_path, index=False)

    # Plain text markdown report
    lines: list[str] = []
    lines.append(f"# Model Evaluation Summary ({market}) — {as_of_date}")
    lines.append("")
    lines.append(f"Rows: {len(df)}")
    lines.append("")

    # Best brier/logloss among rows if present
    if "m_brier" in df.columns:
        b = pd.to_numeric(df["m_brier"], errors="coerce")
        if b.notna().any():
            lines.append(f"Best Brier: {b.min():.6f}")
            lines.append(f"Average Brier: {b.mean():.6f}")
            lines.append("")

    if "m_log_loss" in df.columns:
        ll = pd.to_numeric(df["m_log_loss"], errors="coerce")
        if ll.notna().any():
            lines.append(f"Best LogLoss: {ll.min():.6f}")
            lines.append(f"Average LogLoss: {ll.mean():.6f}")
            lines.append("")

    if "drift_gap_abs_mean" in df.columns:
        d = pd.to_numeric(df["drift_gap_abs_mean"], errors="coerce")
        if d.notna().any():
            lines.append(f"Average |Calibration Gap|: {d.mean():.6f}")
            lines.append("")

    lines.append("Latest Runs:")
    lines.append("")

    preview_cols = [
        c
        for c in [
            "market",
            "event_rule",
            "horizon_days",
            "picked_tag",
            "m_brier",
            "m_log_loss",
            "m_auc",
            "m_ap",
            "drift_gap_last",
            "topk_precision_last",
            "topk_recall_last",
            "topk_f1_last",
            "artifact_dir",
        ]
        if c in df.columns
    ]

    preview_df = df[preview_cols] if preview_cols else df.head(10)
    lines.append(preview_df.to_string(index=False))

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


# -----------------------------
# Phase 8: scan walk-forward backtests
# -----------------------------
def build_backtest_eval_summary(
    market: Optional[str] = None,
    repo_root: Optional[Path] = None,
    latest_only: bool = True,
) -> pd.DataFrame:
    """
    Scans artifacts/backtests/** for fold_metrics_all.csv.
    Produces one row per backtest run (and optionally latest-only per market/tag).
    """
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    bt_dir = repo_root / "artifacts" / "backtests"
    if not bt_dir.exists():
        return pd.DataFrame()

    runs = sorted([p for p in bt_dir.glob("*") if p.is_dir()])

    rows: list[dict] = []
    for run_dir in runs:
        # run_id convention: wf_eventlogit_v1_SPY_YYYYMMDD_HHMMSS
        run_id = run_dir.name

        if market and f"_{market}_" not in run_id and not run_id.endswith(f"_{market}"):
            # still allow if config inside says market
            pass

        fold_path = run_dir / "fold_metrics_all.csv"
        df = _read_csv_if_exists(fold_path)
        if df is None or df.empty:
            continue

        # If market filter provided, enforce via df contents if possible
        if market and "market" in df.columns:
            if not (df["market"].astype(str) == market).any():
                continue

        # Aggregate across folds per horizon
        for h, sub in df.groupby("horizon_days"):
            h = _safe_int(h) or -1

            row = {
                "backtest_run_id": run_id,
                "artifact_dir": str(run_dir),
                "market": market or (sub.get("market", pd.Series([None])).iloc[0] if "market" in sub.columns else None),
                "horizon_days": h,
                "n_folds": int(len(sub)),
                "mean_test_brier_prod": float(pd.to_numeric(sub["test_prod_brier"], errors="coerce").mean()),
                "mean_test_logloss_prod": float(pd.to_numeric(sub["test_prod_log_loss"], errors="coerce").mean()),
                "mean_test_auc_prod": float(pd.to_numeric(sub["test_prod_auc"], errors="coerce").mean()),
                "mean_test_ap_prod": float(pd.to_numeric(sub["test_prod_ap"], errors="coerce").mean()),
                "mean_test_pmean_prod": float(pd.to_numeric(sub["test_prod_p_mean"], errors="coerce").mean()),
                "mean_test_event_rate": float(pd.to_numeric(sub["event_rate_test"], errors="coerce").mean()),
                "chosen_stream_mode": str(sub["chosen_stream"].mode().iloc[0]) if "chosen_stream" in sub.columns and not sub["chosen_stream"].mode().empty else None,
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Latest-only: keep newest run per (market, horizon_days)
    if latest_only:
        # Extract timestamp tail: ..._YYYYMMDD_HHMMSS
        def _ts_key(run_id: str) -> str:
            m = re.search(r"_(\d{8})_(\d{6})$", run_id)
            return m.group(1) + "_" + m.group(2) if m else run_id

        out["_ts"] = out["backtest_run_id"].astype(str).apply(_ts_key)
        out = (
            out.sort_values("_ts", ascending=True)
            .groupby(["market", "horizon_days"], as_index=False)
            .tail(1)
            .drop(columns=["_ts"])
        )

    return out


def write_backtest_outputs(
    df: pd.DataFrame,
    market: str,
    as_of_date: str,
    repo_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    outputs_dir = repo_root / "artifacts" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = outputs_dir / f"backtest_leaderboard_{market}_{as_of_date}.csv"
    md_path = outputs_dir / f"backtest_report_{market}_{as_of_date}.md"

    df.to_csv(csv_path, index=False)

    lines: list[str] = []
    lines.append(f"# Walk-Forward Backtest Summary ({market}) — {as_of_date}")
    lines.append("")
    lines.append(f"Rows: {len(df)}")
    lines.append("")

    # simple best-by-brier per horizon
    if "mean_test_brier_prod" in df.columns:
        b = pd.to_numeric(df["mean_test_brier_prod"], errors="coerce")
        if b.notna().any():
            lines.append(f"Best Mean Test Brier: {b.min():.6f}")
            lines.append("")

    preview_cols = [
        c
        for c in [
            "market",
            "horizon_days",
            "n_folds",
            "mean_test_brier_prod",
            "mean_test_logloss_prod",
            "mean_test_auc_prod",
            "mean_test_ap_prod",
            "mean_test_event_rate",
            "mean_test_pmean_prod",
            "chosen_stream_mode",
            "artifact_dir",
        ]
        if c in df.columns
    ]
    preview_df = df[preview_cols] if preview_cols else df.head(20)
    lines.append(preview_df.to_string(index=False))

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


# -----------------------------
# CLI entry
# -----------------------------
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    outputs_dir = repo_root / "artifacts" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[evaluator] repo_root={repo_root}")

    # Phase 7
    df7 = build_model_eval_summary(market="SPY", repo_root=repo_root, latest_only=True)
    print(f"[evaluator] phase7 model runs rows={len(df7)}")
    if not df7.empty:
        c7, m7 = write_eval_outputs(df7, market="SPY", as_of_date="latest", repo_root=repo_root)
        print(f"[evaluator] wrote: {c7}")
        print(f"[evaluator] wrote: {m7}")

    # Phase 8
    df8 = build_backtest_eval_summary(market="SPY", repo_root=repo_root, latest_only=True)
    print(f"[evaluator] phase8 backtests rows={len(df8)}")
    if not df8.empty:
        c8, m8 = write_backtest_outputs(df8, market="SPY", as_of_date="latest", repo_root=repo_root)
        print(f"[evaluator] wrote: {c8}")
        print(f"[evaluator] wrote: {m8}")

    # python -m src.reporting.model_evaluator
