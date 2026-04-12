# src/targets/targets_builder.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import pandas as pd
from google.cloud import bigquery

from src.core.config import settings
from src.ingest.write_raw import read_raw_series_asof

DEFAULT_TARGET_VERSION = settings.DEFAULT_TARGET_VERSION

TARGETS_TABLE = "targets_v3"
TARGETS_STAGE_TABLE = "targets_stage_v3"
TARGETS_LATEST_VIEW = settings.TARGETS_LATEST_VIEW  # e.g. "targets_latest_v3"

DEFAULT_HORIZONS_DAYS: tuple[int, ...] = (5, 21, 63)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for target write: {missing}")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dates + numerics
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    df["forward_date"] = pd.to_datetime(df["forward_date"]).dt.date
    df["target_value"] = pd.to_numeric(df["target_value"], errors="coerce")
    df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce").astype("Int64")

    # Timestamps (force UTC)
    df["available_time"] = pd.to_datetime(df["available_time"], utc=True)
    df["computed_at"] = pd.to_datetime(df["computed_at"], utc=True)

    # Strings
    for c in ["market", "target_name", "target_version", "run_id"]:
        df[c] = df[c].astype(str)

    return df


def _target_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("market", "STRING"),
        bigquery.SchemaField("as_of_date", "DATE"),
        bigquery.SchemaField("forward_date", "DATE"),
        bigquery.SchemaField("target_name", "STRING"),
        bigquery.SchemaField("target_value", "FLOAT"),
        bigquery.SchemaField("horizon_days", "INT64"),
        bigquery.SchemaField("available_time", "TIMESTAMP"),
        bigquery.SchemaField("target_version", "STRING"),
        bigquery.SchemaField("run_id", "STRING"),
        bigquery.SchemaField("computed_at", "TIMESTAMP"),
    ]


def ensure_table_exists(
    client: bigquery.Client,
    table_id: str,
    schema: list[bigquery.SchemaField],
) -> None:
    """
    Create table if it does not exist.
    """
    try:
        client.get_table(table_id)
    except Exception:
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        print(f"Created table: {table_id}")


def _target_available_time_from_forward_date(forward_date_series: pd.Series) -> pd.Series:
    """
    Conservative availability policy aligned with features:
      available_time = forward_date + 1 day @ 00:00 UTC

    forward_date is the date of P_{t+h} (the close used to compute the forward return).
    """
    fwd = pd.to_datetime(forward_date_series)

    if getattr(fwd.dt, "tz", None) is None:
        fwd = fwd.dt.tz_localize("UTC")
    else:
        fwd = fwd.dt.tz_convert("UTC")

    return fwd + pd.Timedelta(days=1)


def ensure_latest_view(client: bigquery.Client, project: str, dataset: str) -> None:
    """
    Latest view returns the most recently computed row per
    (market, as_of_date, target_name, horizon_days, target_version).
    """
    view_id = f"{project}.{dataset}.{TARGETS_LATEST_VIEW}"
    tgt_id = f"{project}.{dataset}.{TARGETS_TABLE}"

    sql = f"""
    CREATE OR REPLACE VIEW `{view_id}` AS
    SELECT
      market,
      as_of_date,
      forward_date,
      target_name,
      target_value,
      horizon_days,
      available_time,
      target_version,
      run_id,
      computed_at
    FROM `{tgt_id}`
    WHERE target_value IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY market, as_of_date, target_name, horizon_days, target_version
      ORDER BY computed_at DESC
    ) = 1
    """
    client.query(sql, project=client.project).result()


def build_targets_from_prices(
    prices: pd.DataFrame,
    *,
    market: str,
    price_col: str = "value",
    time_col: str = "as_of_date",
    horizons_days: Sequence[int] = DEFAULT_HORIZONS_DAYS,
) -> pd.DataFrame:
    """
    Build forward log return targets:
      fwd_ret_{h}d_log = log(P_{t+h}) - log(P_t)

    Also tracks:
      forward_date = date of P_{t+h}

    Rows at the end without a forward value are dropped (NaN after shift).
    """
    df = prices[[time_col, price_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    df = df[df[price_col] > 0].copy()

    df["logp"] = np.log(df[price_col])

    rows = []
    for h in horizons_days:
        h = int(h)
        name = f"fwd_ret_{h}d_log"

        df[name] = df["logp"].shift(-h) - df["logp"]
        df[f"{name}__forward_date"] = df[time_col].shift(-h)

        tmp = df[[time_col, name, f"{name}__forward_date"]].rename(
            columns={name: "target_value", f"{name}__forward_date": "forward_date"}
        )
        tmp = tmp.dropna(subset=["target_value", "forward_date"]).copy()
        tmp["target_name"] = name
        tmp["horizon_days"] = h
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "market", market)
    out = out.rename(columns={time_col: "as_of_date"})
    return out


def build_and_write_targets_asof(
    *,
    market: str,
    series_id: str,
    as_of_ts: datetime,
    run_id: str,
    target_version: str = DEFAULT_TARGET_VERSION,
    horizons_days: Sequence[int] = DEFAULT_HORIZONS_DAYS,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    client = bigquery.Client(project=settings.GCP_PROJECT_ID)
    project = settings.GCP_PROJECT_ID
    dataset = settings.BQ_DATASET

    raw = read_raw_series_asof(series_id=series_id, as_of_ts=as_of_ts)
    if raw.empty:
        raise ValueError(f"No raw rows returned for series_id={series_id} as_of_ts={as_of_ts}.")

    if start_date is not None:
        sd = pd.to_datetime(start_date).date()
        raw = raw[raw["as_of_date"] >= sd].copy()
    if end_date is not None:
        ed = pd.to_datetime(end_date).date()
        raw = raw[raw["as_of_date"] <= ed].copy()
    if raw.empty:
        raise ValueError("Raw data became empty after filtering.")

    targets = build_targets_from_prices(
        raw,
        market=market,
        price_col="value",
        time_col="as_of_date",
        horizons_days=horizons_days,
    )

    computed_at = _utc_now()
    targets["target_version"] = str(target_version)
    targets["run_id"] = str(run_id)
    targets["computed_at"] = computed_at
    targets["available_time"] = _target_available_time_from_forward_date(targets["forward_date"])

    required = [
        "market",
        "as_of_date",
        "forward_date",
        "target_name",
        "target_value",
        "horizon_days",
        "available_time",
        "target_version",
        "run_id",
        "computed_at",
    ]
    _require_columns(targets, required)
    targets = targets[required].copy()
    targets = _coerce_types(targets)

    stage_table = f"{project}.{dataset}.{TARGETS_STAGE_TABLE}"
    target_table = f"{project}.{dataset}.{TARGETS_TABLE}"
    schema = _target_schema()

    ensure_table_exists(client, stage_table, schema)
    ensure_table_exists(client, target_table, schema)

    cfg = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", schema=schema)
    client.load_table_from_dataframe(targets, stage_table, job_config=cfg).result()

    insert_sql = f"""
    INSERT INTO `{target_table}` (
      market, as_of_date, forward_date, target_name, target_value,
      horizon_days, available_time, target_version, run_id, computed_at
    )
    SELECT
      market, as_of_date, forward_date, target_name, target_value,
      horizon_days, available_time, target_version, run_id, computed_at
    FROM `{stage_table}`
    """
    client.query(insert_sql, project=client.project).result()

    ensure_latest_view(client, project, dataset)
    return targets

