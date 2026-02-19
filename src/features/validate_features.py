# src/features/validate_features.py
from __future__ import annotations

from collections.abc import Iterable
from google.cloud import bigquery

from src.core.config import settings

FEATURES_LATEST_VIEW = "features_latest_v3"


def _as_list(x: str | Iterable[str] | None) -> list[str] | None:
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return [str(v) for v in x]


def _get_markets_for_version(
    client: bigquery.Client,
    view_id: str,
    *,
    feature_version: str,
) -> list[str]:
    sql = f"""
    SELECT DISTINCT market
    FROM `{view_id}`
    WHERE feature_version = @feature_version
    ORDER BY market
    """
    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("feature_version", "STRING", feature_version),
        ]
    )
    df = client.query(sql, job_config=job_cfg).to_dataframe()
    return df["market"].astype(str).tolist()


def validate_features_latest(
    *,
    markets: str | Iterable[str] | None = "SPY",
    feature_version: str = "v2",
) -> None:
    """
    Validate rows in features_latest_v3.

    Parameters
    ----------
    markets:
        - "SPY" (default): validate one market
        - ["SPY","QQQ",...]: validate a list
        - None: validate ALL markets found in the view for feature_version
    feature_version:
        Which feature version to validate (e.g. "v2")
    """
    client = bigquery.Client(project=settings.GCP_PROJECT_ID)
    project = settings.GCP_PROJECT_ID
    dataset = settings.BQ_DATASET
    view_id = f"{project}.{dataset}.{FEATURES_LATEST_VIEW}"

    # Resolve markets
    mkts = _as_list(markets)
    if mkts is None:
        mkts = _get_markets_for_version(client, view_id, feature_version=feature_version)
        if not mkts:
            raise ValueError(
                f"FAIL: no markets found in {FEATURES_LATEST_VIEW} for feature_version={feature_version}"
            )

    print("Validating:", view_id)
    print("Markets:", mkts, "| Version:", feature_version)

    for market in mkts:
        print("\n" + "=" * 90)
        print("Market:", market, "| Version:", feature_version)

        job_cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("market", "STRING", market),
                bigquery.ScalarQueryParameter("feature_version", "STRING", feature_version),
            ]
        )

        # 1) Rowcount
        sql_count = f"""
        SELECT COUNT(*) AS n
        FROM `{view_id}`
        WHERE market = @market
          AND feature_version = @feature_version
        """
        n = client.query(sql_count, job_config=job_cfg).to_dataframe()["n"][0]
        if n == 0:
            raise ValueError(
                f"FAIL: features_latest_v3 returned 0 rows for market={market}, version={feature_version}."
            )
        print(f"PASS: rowcount = {n}")

        # 2) Uniqueness check (should be 0)
        sql_dupes = f"""
        SELECT COUNT(*) AS dupes
        FROM (
          SELECT market, as_of_date, feature_name, feature_version, COUNT(*) AS c
          FROM `{view_id}`
          WHERE market = @market
            AND feature_version = @feature_version
          GROUP BY 1,2,3,4
          HAVING c > 1
        )
        """
        dupes = client.query(sql_dupes, job_config=job_cfg).to_dataframe()["dupes"][0]
        if dupes != 0:
            raise ValueError(f"FAIL: found {dupes} duplicate keys in latest view for market={market}.")
        print("PASS: uniqueness (no duplicate keys)")

        # 3) Invariants
        sql_inv = f"""
        SELECT
          SUM(CASE WHEN feature_name IN ('rv_5d','rv_21d','rv_63d') AND feature_value < 0 THEN 1 ELSE 0 END) AS neg_vol,
          SUM(CASE WHEN feature_name IN ('dd_21','dd_63') AND feature_value > 0 THEN 1 ELSE 0 END) AS pos_dd
        FROM `{view_id}`
        WHERE market = @market
          AND feature_version = @feature_version
        """
        inv = client.query(sql_inv, job_config=job_cfg).to_dataframe().to_dict(orient="records")[0]

        if inv["neg_vol"] != 0:
            raise ValueError(f"FAIL: negative realized vol rows found: {inv['neg_vol']} (market={market})")
        print("PASS: rv_5d/rv_21d/rv_63d non-negative (ignoring NULLs)")

        if inv["pos_dd"] != 0:
            raise ValueError(f"FAIL: positive drawdown rows found: {inv['pos_dd']} (market={market})")
        print("PASS: dd_21/dd_63 <= 0 (ignoring NULLs)")

        # 4) Coverage report (% NULL by feature)
        sql_coverage = f"""
        SELECT
          feature_name,
          COUNT(*) AS n,
          SUM(CASE WHEN feature_value IS NULL THEN 1 ELSE 0 END) AS n_null,
          SAFE_DIVIDE(SUM(CASE WHEN feature_value IS NULL THEN 1 ELSE 0 END), COUNT(*)) AS null_rate
        FROM `{view_id}`
        WHERE market = @market
          AND feature_version = @feature_version
        GROUP BY feature_name
        ORDER BY feature_name
        """
        cov = client.query(sql_coverage, job_config=job_cfg).to_dataframe()
        print("\nCoverage by feature (NULL rate expected early in series for rolling windows):")
        print(cov.to_string(index=False))

    print("\nFeature validation complete.")


if __name__ == "__main__":
    validate_features_latest()

    # Examples:
    # python -m src.features.validate_features
    # python -c "from src.features.validate_features import validate_features_latest; validate_features_latest(markets=['SPY','QQQ'], feature_version='v2')"
    # python -c "from src.features.validate_features import validate_features_latest; validate_features_latest(markets=None, feature_version='v2')"
