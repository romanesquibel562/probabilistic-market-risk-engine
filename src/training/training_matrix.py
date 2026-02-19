# src/training/training_matrix.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from google.cloud import bigquery

from src.core.config import settings


# View names are centralized in Settings (these are the "latest" consolidated views).
FEATURES_VIEW = settings.FEATURES_LATEST_VIEW
TARGETS_VIEW = settings.TARGETS_LATEST_VIEW


@dataclass(frozen=True)
class SplitConfig:
    train_end: date
    test_start: date
    test_end: date | None = None


def _bq_client() -> bigquery.Client:
    return bigquery.Client(project=settings.GCP_PROJECT_ID)


def _validate_target_pair(target_name: str, horizon_days: int) -> None:
    """
    Only enforces pairing rules for forward-log-return targets.
    Other target families (drawdown events, vol-expansion events, etc.)
    bypass this check until we add family-specific validators.
    """
    expected = f"fwd_ret_{horizon_days}d_log"
    if target_name.startswith("fwd_ret_") and target_name.endswith("_log") and target_name != expected:
        raise ValueError(
            f"target_name='{target_name}' does not match horizon_days={horizon_days}. "
            f"Expected '{expected}'."
        )


def load_features_wide(
    *,
    market: str,
    feature_version: str = settings.DEFAULT_FEATURE_VERSION,
    start_date: str | None = None,
    end_date: str | None = None,
    strict_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Load features from the latest features view and pivot to wide.

    Output columns:
      market, as_of_date, <feature columns...>
    """
    client = _bq_client()
    project = settings.GCP_PROJECT_ID
    dataset = settings.BQ_DATASET
    view_id = f"{project}.{dataset}.{FEATURES_VIEW}"

    where = ["market = @market", "feature_version = @feature_version"]
    params: list[bigquery.ScalarQueryParameter] = [
        bigquery.ScalarQueryParameter("market", "STRING", market),
        bigquery.ScalarQueryParameter("feature_version", "STRING", feature_version),
    ]

    if start_date:
        where.append("as_of_date >= @start_date")
        params.append(bigquery.ScalarQueryParameter("start_date", "DATE", pd.to_datetime(start_date).date()))
    if end_date:
        where.append("as_of_date <= @end_date")
        params.append(bigquery.ScalarQueryParameter("end_date", "DATE", pd.to_datetime(end_date).date()))

    sql = f"""
    SELECT
        market,
        as_of_date,
        feature_name,
        feature_value
    FROM `{view_id}`
    WHERE {" AND ".join(where)}
    ORDER BY as_of_date, feature_name
    """

    df_long = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=params),
    ).to_dataframe()

    if df_long.empty:
        return pd.DataFrame(columns=["market", "as_of_date"])

    # Optional: fail loudly if the latest view still contains duplicates
    # for the same (market, as_of_date, feature_name). "first" in pivot
    # can silently mask these issues.
    if strict_duplicates:
        dup_mask = df_long.duplicated(subset=["market", "as_of_date", "feature_name"], keep=False)
        if dup_mask.any():
            sample = (
                df_long.loc[dup_mask, ["market", "as_of_date", "feature_name"]]
                .drop_duplicates()
                .head(20)
            )
            raise ValueError(
                "Duplicate feature rows detected for the same (market, as_of_date, feature_name). "
                "This should not happen in a latest view. Sample:\n"
                f"{sample.to_string(index=False)}"
            )

    # Pivot to wide format
    wide = (
        df_long.pivot_table(
            index=["market", "as_of_date"],
            columns="feature_name",
            values="feature_value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide


def load_targets(
    *,
    market: str,
    target_name: str,
    horizon_days: int,
    target_version: str = settings.DEFAULT_TARGET_VERSION,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load a single target series from the latest targets view.

    Output columns:
      market, as_of_date, target_value
    """
    client = _bq_client()
    project = settings.GCP_PROJECT_ID
    dataset = settings.BQ_DATASET
    view_id = f"{project}.{dataset}.{TARGETS_VIEW}"

    where = [
        "market = @market",
        "target_version = @target_version",
        "target_name = @target_name",
        "horizon_days = @horizon_days",
    ]
    params: list[bigquery.ScalarQueryParameter] = [
        bigquery.ScalarQueryParameter("market", "STRING", market),
        bigquery.ScalarQueryParameter("target_version", "STRING", target_version),
        bigquery.ScalarQueryParameter("target_name", "STRING", target_name),
        bigquery.ScalarQueryParameter("horizon_days", "INT64", horizon_days),
    ]

    if start_date:
        where.append("as_of_date >= @start_date")
        params.append(bigquery.ScalarQueryParameter("start_date", "DATE", pd.to_datetime(start_date).date()))
    if end_date:
        where.append("as_of_date <= @end_date")
        params.append(bigquery.ScalarQueryParameter("end_date", "DATE", pd.to_datetime(end_date).date()))

    sql = f"""
    SELECT market, as_of_date, target_value
    FROM `{view_id}`
    WHERE {" AND ".join(where)}
    ORDER BY as_of_date
    """

    return client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=params),
    ).to_dataframe()


def build_training_matrix(
    *,
    market: str,
    feature_version: str = settings.DEFAULT_FEATURE_VERSION,
    target_name: str = "fwd_ret_5d_log",
    horizon_days: int = 5,
    target_version: str = settings.DEFAULT_TARGET_VERSION,
    start_date: str | None = None,
    end_date: str | None = None,
    dropna_features: bool = True,
    strict_feature_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Join wide features to target series on (market, as_of_date).

    - Drops rows where target is NULL (end-of-series forward shift).
    - Optionally drops rows with NULL feature values (warmup windows).
    """
    _validate_target_pair(target_name, horizon_days)

    X = load_features_wide(
        market=market,
        feature_version=feature_version,
        start_date=start_date,
        end_date=end_date,
        strict_duplicates=strict_feature_duplicates,
    )
    y = load_targets(
        market=market,
        target_name=target_name,
        horizon_days=horizon_days,
        target_version=target_version,
        start_date=start_date,
        end_date=end_date,
    )

    if X.empty or y.empty:
        # Return a consistent schema even when empty.
        cols = ["market", "as_of_date", "target_value"]
        if not X.empty:
            cols = list(X.columns) + ["target_value"]
        return pd.DataFrame(columns=cols)

    df = X.merge(y, on=["market", "as_of_date"], how="inner")
    df = df.dropna(subset=["target_value"]).reset_index(drop=True)

    if dropna_features and not df.empty:
        feature_cols = [c for c in df.columns if c not in ("market", "as_of_date", "target_value")]
        if feature_cols:
            df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


def split_train_test(df: pd.DataFrame, split: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split (no shuffling).
    """
    df = df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date

    train = df[df["as_of_date"] <= split.train_end].copy()
    test = df[df["as_of_date"] >= split.test_start].copy()

    if split.test_end is not None:
        test = test[test["as_of_date"] <= split.test_end].copy()

    return train.reset_index(drop=True), test.reset_index(drop=True)

