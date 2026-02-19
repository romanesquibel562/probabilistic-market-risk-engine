# src/ingest/write_raw.py
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from google.cloud import bigquery

from src.core.config import settings


RAW_TABLE = "raw_series_values_v3"
STAGE_TABLE = "raw_series_values_stage_v3"
LATEST_VIEW = "raw_series_values_latest_v3"


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for raw ingest: {missing}")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strings
    df["series_id"] = df["series_id"].astype(str)
    df["source"] = df["source"].astype(str)

    # Dates + numerics
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce").dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Force UTC tz-aware timestamps for BigQuery TIMESTAMP
    df["available_time"] = pd.to_datetime(df["available_time"], utc=True, errors="coerce")
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True, errors="coerce")

    df = df.dropna(
        subset=[
            "series_id",
            "source",
            "as_of_date",
            "value",
            "available_time",
            "ingested_at",
        ]
    )
    return df


def ensure_latest_view(client: bigquery.Client, project: str, dataset: str) -> None:
    """
    Create/update a view that returns the latest ingested row per (series_id, source, as_of_date).
    """
    view_id = f"{project}.{dataset}.{LATEST_VIEW}"
    raw_id = f"{project}.{dataset}.{RAW_TABLE}"

    sql = f"""
    CREATE OR REPLACE VIEW `{view_id}` AS
    SELECT
      series_id,
      source,
      as_of_date,
      value,
      available_time,
      ingested_at
    FROM `{raw_id}`
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY series_id, source, as_of_date
      ORDER BY ingested_at DESC
    ) = 1
    """
    client.query(sql, project=client.project).result()


def upsert_raw_series(df: pd.DataFrame) -> None:
    """
    Production-safe raw ingest:
      1) WRITE_TRUNCATE into stage table (current batch)
      2) INSERT stage -> raw (append-only)
      3) Maintain a deduping view (latest per key)
    """
    client = bigquery.Client(project=settings.GCP_PROJECT_ID)
    project = settings.GCP_PROJECT_ID
    dataset = settings.BQ_DATASET

    required = ["series_id", "source", "as_of_date", "value", "available_time", "ingested_at"]
    _require_columns(df, required)
    df = _coerce_types(df)

    stage_table = f"{project}.{dataset}.{STAGE_TABLE}"
    raw_table = f"{project}.{dataset}.{RAW_TABLE}"

    schema = [
        bigquery.SchemaField("series_id", "STRING"),
        bigquery.SchemaField("source", "STRING"),
        bigquery.SchemaField("as_of_date", "DATE"),
        bigquery.SchemaField("value", "FLOAT"),
        bigquery.SchemaField("available_time", "TIMESTAMP"),
        bigquery.SchemaField("ingested_at", "TIMESTAMP"),
    ]

    # 1) Load batch into staging
    stage_cfg = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", schema=schema)
    client.load_table_from_dataframe(df, stage_table, job_config=stage_cfg).result()

    # Debug (keep for now; you can remove later)
    stage_check_sql = f"""
    SELECT COUNT(*) AS n, MIN(as_of_date) AS mind, MAX(as_of_date) AS maxd
    FROM `{stage_table}`
    """
    stage_check = (
        client.query(stage_check_sql, project=client.project)
        .to_dataframe()
        .to_dict(orient="records")[0]
    )
    print("STAGE CHECK:", stage_check)

    # 2) Append stage into raw (no MERGE / no partition DML issues)
    insert_sql = f"""
    INSERT INTO `{raw_table}` (series_id, source, as_of_date, value, available_time, ingested_at)
    SELECT series_id, source, as_of_date, value, available_time, ingested_at
    FROM `{stage_table}`
    """
    client.query(insert_sql, project=client.project).result()

    # 3) Ensure latest view exists (idempotent at read time)
    ensure_latest_view(client, project, dataset)

    # Target check: count in view is the "effective" deduped set
    view_table = f"{project}.{dataset}.{LATEST_VIEW}"
    target_check_sql = f"""
    SELECT COUNT(*) AS n, MIN(as_of_date) AS mind, MAX(as_of_date) AS maxd
    FROM `{view_table}`
    """
    target_check = (
        client.query(target_check_sql, project=client.project)
        .to_dataframe()
        .to_dict(orient="records")[0]
    )
    print("LATEST VIEW CHECK:", target_check)


def read_raw_series_asof(series_id: str, as_of_ts: datetime) -> pd.DataFrame:
    """
    Anti-leakage read:
      - reads from latest view (deduped)
      - enforces available_time <= as_of_ts
    """
    if as_of_ts.tzinfo is None:
        # Treat naive times as UTC to avoid silent local-time leakage bugs
        as_of_ts = as_of_ts.replace(tzinfo=timezone.utc)
    else:
        as_of_ts = as_of_ts.astimezone(timezone.utc)

    client = bigquery.Client(project=settings.GCP_PROJECT_ID)
    project = settings.GCP_PROJECT_ID
    dataset = settings.BQ_DATASET

    view_id = f"{project}.{dataset}.{LATEST_VIEW}"

    sql = f"""
    SELECT series_id, source, as_of_date, value, available_time, ingested_at
    FROM `{view_id}`
    WHERE series_id = @series_id
      AND available_time <= @as_of_ts
    ORDER BY as_of_date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("series_id", "STRING", series_id),
            bigquery.ScalarQueryParameter("as_of_ts", "TIMESTAMP", as_of_ts),
        ]
    )
    return client.query(sql, job_config=job_config, project=client.project).to_dataframe()

