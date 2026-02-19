# src/features/feature_builder.py
from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Iterable

import pandas as pd
from google.cloud import bigquery

from src.core.config import settings
from src.ingest.write_raw import read_raw_series_asof
from src.features.market_router import compute_features_for_market

FEATURE_VERSION = "v2"

FEATURES_TABLE = "features_v3"
FEATURES_STAGE_TABLE = "features_stage_v3"
FEATURES_LATEST_VIEW = "features_latest_v3"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature write: {missing}")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # as_of_date stored as DATE in BigQuery
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date

    # numeric values; NaNs become NULLs in BQ
    df["feature_value"] = pd.to_numeric(df["feature_value"], errors="coerce")

    # TIMESTAMP columns must be tz-aware UTC
    df["available_time"] = pd.to_datetime(df["available_time"], utc=True)
    df["computed_at"] = pd.to_datetime(df["computed_at"], utc=True)

    # Strings
    for c in ["market", "feature_name", "feature_version", "run_id"]:
        df[c] = df[c].astype(str)

    return df


def _compute_available_time(as_of_date_series: pd.Series) -> pd.Series:
    """
    Conservative v1 availability policy (anti-leakage):
      available_time = as_of_date + 1 day @ 00:00 UTC

    Meaning: features derived from a day's close are only available the next day.
    Later we can replace with a trading-calendar-aware rule.
    """
    dt_utc = pd.to_datetime(as_of_date_series).dt.tz_localize("UTC")
    return dt_utc + pd.Timedelta(days=1)


def ensure_latest_view(client: bigquery.Client, project: str, dataset: str) -> None:
    """
    Create/update a view that returns the latest computed row per
    (market, as_of_date, feature_name, feature_version).
    """
    view_id = f"{project}.{dataset}.{FEATURES_LATEST_VIEW}"
    feat_id = f"{project}.{dataset}.{FEATURES_TABLE}"

    sql = f"""
    CREATE OR REPLACE VIEW `{view_id}` AS
    SELECT
      market,
      as_of_date,
      feature_name,
      feature_value,
      available_time,
      feature_version,
      run_id,
      computed_at
    FROM `{feat_id}`
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY market, as_of_date, feature_name, feature_version
      ORDER BY computed_at DESC
    ) = 1
    """
    client.query(sql, project=client.project).result()

def build_and_write_features_many_asof(  # <-- new
    *,
    markets: Iterable[tuple[str, str]],
    as_of_ts: datetime,
    run_id: str,
    feature_version: str = FEATURE_VERSION,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Multi-market wrapper for build_and_write_features_asof.

    Parameters
    ----------
    markets:
        Iterable of (market, series_id) pairs, e.g.
        [("SPY","mkt.spy_close"), ("QQQ","mkt.qqq_close")]

    Returns
    -------
    pd.DataFrame
        Concatenated features across all markets in the same long schema.
    """
    out: list[pd.DataFrame] = []

    for market, series_id in markets:
        df = build_and_write_features_asof(
            market=market,
            series_id=series_id,
            as_of_ts=as_of_ts,
            run_id=run_id,
            feature_version=feature_version,
            start_date=start_date,
            end_date=end_date,
        )
        out.append(df)

    if not out:
        return pd.DataFrame(
            columns=[
                "market",
                "as_of_date",
                "feature_name",
                "feature_value",
                "available_time",
                "feature_version",
                "run_id",
                "computed_at",
            ]
        )

    return pd.concat(out, ignore_index=True)



def build_and_write_features_asof(
    *,
    market: str,
    series_id: str,
    as_of_ts: datetime,
    run_id: str,
    feature_version: str = FEATURE_VERSION,
    start_date: str | None = None,   # "YYYY-MM-DD"
    end_date: str | None = None,     # "YYYY-MM-DD"
) -> pd.DataFrame:
    """
    End-to-end Step 2 with optional date filters.

    start_date/end_date filter the raw history BEFORE feature computation.
    Useful for fast dev runs and controlled backfills.
    """
    client = bigquery.Client(project=settings.GCP_PROJECT_ID)
    project = settings.GCP_PROJECT_ID
    dataset = settings.BQ_DATASET

    # 1) leakage-safe read from raw latest view
    raw = read_raw_series_asof(series_id=series_id, as_of_ts=as_of_ts)

    if raw.empty:
        raise ValueError(
            f"No raw rows returned for series_id={series_id} as_of_ts={as_of_ts}."
        )

    # Optional filter window (DATE-grain)
    if start_date is not None:
        sd = pd.to_datetime(start_date).date()
        raw = raw[raw["as_of_date"] >= sd].copy()

    if end_date is not None:
        ed = pd.to_datetime(end_date).date()
        raw = raw[raw["as_of_date"] <= ed].copy()

    if raw.empty:
        raise ValueError(
            f"Raw data became empty after filtering. start_date={start_date}, end_date={end_date}."
        )

    # 2) compute features using the raw schema: as_of_date + value
    feats = compute_features_for_market(
        market=market,
        prices=raw,
        price_col="value",
        time_col="as_of_date",
    )

    # 3) stamp metadata
    feats["feature_version"] = feature_version
    feats["run_id"] = run_id
    feats["computed_at"] = _utc_now()
    feats["available_time"] = _compute_available_time(feats["as_of_date"])

    required = [
        "market",
        "as_of_date",
        "feature_name",
        "feature_value",
        "available_time",
        "feature_version",
        "run_id",
        "computed_at",
    ]
    _require_columns(feats, required)
    feats = feats[required].copy()
    feats = _coerce_types(feats)

    stage_table = f"{project}.{dataset}.{FEATURES_STAGE_TABLE}"
    target_table = f"{project}.{dataset}.{FEATURES_TABLE}"

    schema = [
        bigquery.SchemaField("market", "STRING"),
        bigquery.SchemaField("as_of_date", "DATE"),
        bigquery.SchemaField("feature_name", "STRING"),
        bigquery.SchemaField("feature_value", "FLOAT"),
        bigquery.SchemaField("available_time", "TIMESTAMP"),
        bigquery.SchemaField("feature_version", "STRING"),
        bigquery.SchemaField("run_id", "STRING"),
        bigquery.SchemaField("computed_at", "TIMESTAMP"),
    ]

    # 4) load batch to stage
    stage_cfg = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", schema=schema)
    client.load_table_from_dataframe(feats, stage_table, job_config=stage_cfg).result()

    # 5) UPSERT stage into target (MERGE)
    merge_sql = f"""
    MERGE `{target_table}` T
    USING `{stage_table}` S
    ON
      T.market = S.market
      AND T.as_of_date = S.as_of_date
      AND T.feature_name = S.feature_name
      AND T.feature_version = S.feature_version
    WHEN MATCHED THEN
      UPDATE SET
        feature_value = S.feature_value,
        available_time = S.available_time,
        run_id = S.run_id,
        computed_at = S.computed_at
    WHEN NOT MATCHED THEN
      INSERT (
        market, as_of_date, feature_name, feature_value,
        available_time, feature_version, run_id, computed_at
      )
      VALUES (
        S.market, S.as_of_date, S.feature_name, S.feature_value,
        S.available_time, S.feature_version, S.run_id, S.computed_at
      )
    """
    client.query(merge_sql, project=client.project).result()

    # 6) maintain latest view
    ensure_latest_view(client, project, dataset)

    return feats
