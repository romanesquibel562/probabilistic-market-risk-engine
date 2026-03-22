# src/targets/validate_targets.py
from __future__ import annotations

from google.cloud import bigquery

from src.core.config import settings


TARGETS_LATEST_VIEW = settings.TARGETS_LATEST_VIEW
DEFAULT_EXPECTED_HORIZONS = [5, 21, 63]


def validate_targets_latest(
    market: str = "SPY",
    target_version: str | None = None,
    expected_horizons: list[int] | None = None,
) -> None:
    """
    Validates the targets_latest view for a market + target_version.

    Checks:
      1) rowcount > 0
      2) expected horizons present
      3) uniqueness of (market, as_of_date, target_name, horizon_days, target_version)
      4) forward_date sanity (>= as_of_date)
      5) available_time sanity + exact equality check (forward-date based)
      6) coverage report (null rate per target/horizon)
    """
    if target_version is None:
        target_version = settings.DEFAULT_TARGET_VERSION
    if expected_horizons is None:
        expected_horizons = DEFAULT_EXPECTED_HORIZONS

    client = bigquery.Client(project=settings.GCP_PROJECT_ID)
    project = settings.GCP_PROJECT_ID
    dataset = settings.BQ_DATASET
    view_id = f"{project}.{dataset}.{TARGETS_LATEST_VIEW}"

    print("Validating:", view_id)
    print("Market:", market, "| Version:", target_version)

    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("market", "STRING", market),
            bigquery.ScalarQueryParameter("target_version", "STRING", target_version),
        ]
    )

    # 1) Rowcount check
    sql_count = f"""
    SELECT COUNT(*) AS n
    FROM `{view_id}`
    WHERE market = @market
      AND target_version = @target_version
    """
    n = int(client.query(sql_count, job_config=job_cfg).to_dataframe()["n"][0])
    if n == 0:
        raise ValueError("FAIL: targets_latest returned 0 rows for this market/version.")
    print(f"PASS: rowcount = {n}")

    # 1b) Horizons presence check
    sql_horiz = f"""
    SELECT ARRAY_AGG(DISTINCT horizon_days ORDER BY horizon_days) AS horizons
    FROM `{view_id}`
    WHERE market = @market
      AND target_version = @target_version
    """
    horizons = client.query(sql_horiz, job_config=job_cfg).to_dataframe()["horizons"][0]
    horizons_list = list(horizons) if horizons is not None else []

    if horizons_list != expected_horizons:
        raise ValueError(f"FAIL: horizons present {horizons_list}, expected {expected_horizons}.")
    print(f"PASS: horizons present = {horizons_list}")

    # 2) Uniqueness check (should be 0)
    sql_dupes = f"""
    SELECT COUNT(*) AS dupes
    FROM (
      SELECT market, as_of_date, target_name, horizon_days, target_version, COUNT(*) AS c
      FROM `{view_id}`
      WHERE market = @market
        AND target_version = @target_version
      GROUP BY 1,2,3,4,5
      HAVING c > 1
    )
    """
    dupes = int(client.query(sql_dupes, job_config=job_cfg).to_dataframe()["dupes"][0])
    if dupes != 0:
        raise ValueError(f"FAIL: found {dupes} duplicate keys in targets latest view.")
    print("PASS: uniqueness (no duplicate keys)")

    # 3) forward_date sanity
    # forward_date should never be earlier than as_of_date
    sql_fwd = f"""
    SELECT
      SUM(CASE WHEN forward_date IS NULL THEN 1 ELSE 0 END) AS null_fwd,
      SUM(CASE WHEN forward_date < as_of_date THEN 1 ELSE 0 END) AS bad_fwd
    FROM `{view_id}`
    WHERE market = @market
      AND target_version = @target_version
    """
    fwd = client.query(sql_fwd, job_config=job_cfg).to_dataframe().to_dict(orient="records")[0]
    null_fwd = int(fwd["null_fwd"])
    bad_fwd = int(fwd["bad_fwd"])

    if null_fwd != 0:
        raise ValueError(
            f"FAIL: found {null_fwd} rows with NULL forward_date. "
            "Re-run targets with the updated builder that writes forward_date."
        )
    if bad_fwd != 0:
        raise ValueError(f"FAIL: found {bad_fwd} rows where forward_date < as_of_date.")
    print("PASS: forward_date present and >= as_of_date")

    # 4) Availability gate sanity (weak check)
    # available_time should not be earlier than forward_date (and certainly not earlier than as_of_date)
    sql_avail = f"""
    SELECT
      SUM(CASE WHEN available_time < TIMESTAMP(as_of_date) THEN 1 ELSE 0 END) AS bad_vs_asof,
      SUM(CASE WHEN available_time < TIMESTAMP(forward_date) THEN 1 ELSE 0 END) AS bad_vs_fwd
    FROM `{view_id}`
    WHERE market = @market
      AND target_version = @target_version
    """
    av = client.query(sql_avail, job_config=job_cfg).to_dataframe().to_dict(orient="records")[0]
    bad_vs_asof = int(av["bad_vs_asof"])
    bad_vs_fwd = int(av["bad_vs_fwd"])

    if bad_vs_asof != 0:
        raise ValueError(f"FAIL: found {bad_vs_asof} rows where available_time < as_of_date.")
    if bad_vs_fwd != 0:
        raise ValueError(f"FAIL: found {bad_vs_fwd} rows where available_time < forward_date.")
    print("PASS: available_time not earlier than as_of_date/forward_date")

    # 4b) Availability exact check (strong check)
    # New invariant (aligned with features' conservative policy):
    #   available_time == forward_date + 1 day @ 00:00 UTC
    sql_avail_exact = f"""
    SELECT
      SUM(
        CASE
          WHEN available_time != TIMESTAMP(DATE_ADD(forward_date, INTERVAL 1 DAY))
          THEN 1 ELSE 0
        END
      ) AS bad_exact
    FROM `{view_id}`
    WHERE market = @market
      AND target_version = @target_version
    """
    bad_exact = int(client.query(sql_avail_exact, job_config=job_cfg).to_dataframe()["bad_exact"][0])
    if bad_exact != 0:
        raise ValueError(
            f"FAIL: found {bad_exact} rows where available_time != forward_date + 1 day (UTC midnight)."
        )
    print("PASS: available_time equals forward_date + 1 day (UTC midnight)")

    # 5) Coverage report (NULL rate per horizon/target)
    sql_cov = f"""
    SELECT
      target_name,
      horizon_days,
      COUNT(*) AS n,
      SUM(CASE WHEN target_value IS NULL THEN 1 ELSE 0 END) AS n_null,
      SAFE_DIVIDE(SUM(CASE WHEN target_value IS NULL THEN 1 ELSE 0 END), COUNT(*)) AS null_rate
    FROM `{view_id}`
    WHERE market = @market
      AND target_version = @target_version
    GROUP BY target_name, horizon_days
    ORDER BY target_name, horizon_days
    """
    cov = client.query(sql_cov, job_config=job_cfg).to_dataframe()
    print("\nCoverage by target/horizon:")
    print(cov.to_string(index=False))

    total_nulls = int(cov["n_null"].sum()) if not cov.empty else 0
    if total_nulls != 0:
        print(f"WARNING: found {total_nulls} NULL target_value rows (likely legacy rows from older runs).")
        print("         This should go to ~0 after re-running targets over your full lookback with the updated builder.")
    else:
        print("PASS: no NULL target_value rows")


if __name__ == "__main__":
    validate_targets_latest()

    # Run:
    #   python -m src.targets.validate_targets



    # Run:
    #   python -m src.targets.validate_targets


