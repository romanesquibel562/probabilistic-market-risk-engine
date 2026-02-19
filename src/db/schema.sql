-- =========================
-- Dataset: {{PROJECT}}.{{DATASET}}
-- Core tables (MVP)
-- =========================

CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.run_log` (
  run_id STRING,
  run_ts TIMESTAMP,
  as_of_date DATE,
  env STRING,
  git_sha STRING,
  notes STRING
)
PARTITION BY as_of_date
CLUSTER BY run_id;

CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.raw_series_values` (
  series_id STRING,
  source STRING,
  observation_date DATE,
  value FLOAT64,
  ingest_time TIMESTAMP,
  available_time TIMESTAMP
)
PARTITION BY observation_date
CLUSTER BY series_id;

-- v2 raw tables (anti-leakage naming: as_of_date, ingested_at)
CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.raw_series_values_v2` (
  series_id STRING,
  source STRING,
  as_of_date DATE,
  value FLOAT64,
  available_time TIMESTAMP,
  ingested_at TIMESTAMP
)
PARTITION BY as_of_date
CLUSTER BY series_id;

CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.raw_series_values_stage_v2` (
  series_id STRING,
  source STRING,
  as_of_date DATE,
  value FLOAT64,
  available_time TIMESTAMP,
  ingested_at TIMESTAMP
);
-- v3 (unpartitioned) raw tables to avoid sandbox partition expiration limits
CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.raw_series_values_v3` (
  series_id STRING,
  source STRING,
  as_of_date DATE,
  value FLOAT64,
  available_time TIMESTAMP,
  ingested_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.raw_series_values_stage_v3` (
  series_id STRING,
  source STRING,
  as_of_date DATE,
  value FLOAT64,
  available_time TIMESTAMP,
  ingested_at TIMESTAMP
);

-- v3 (unpartitioned) features tables to match raw v3 approach
CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.features_v3` (
  market STRING,
  as_of_date DATE,
  feature_name STRING,
  feature_value FLOAT64,
  available_time TIMESTAMP,
  feature_version STRING,
  run_id STRING,
  computed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.features_stage_v3` (
  market STRING,
  as_of_date DATE,
  feature_name STRING,
  feature_value FLOAT64,
  available_time TIMESTAMP,
  feature_version STRING,
  run_id STRING,
  computed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.features` (
  market STRING,
  feature_id STRING,
  feature_date DATE,
  value FLOAT64,
  run_id STRING
)
PARTITION BY feature_date
CLUSTER BY market, feature_id;

-- v3 (unpartitioned) targets tables to match raw/features v3 approach
CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.targets_v3` (
  market STRING,
  as_of_date DATE,
  target_name STRING,
  target_value FLOAT64,
  horizon_days INT64,
  available_time TIMESTAMP,
  target_version STRING,
  run_id STRING,
  computed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.targets_stage_v3` (
  market STRING,
  as_of_date DATE,
  target_name STRING,
  target_value FLOAT64,
  horizon_days INT64,
  available_time TIMESTAMP,
  target_version STRING,
  run_id STRING,
  computed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.predictions` (
  market STRING,
  horizon INT64,
  target STRING,
  prediction_date DATE,
  probability FLOAT64,
  model STRING,
  calibrated BOOL,
  run_id STRING
)
PARTITION BY prediction_date
CLUSTER BY market, horizon, target;

CREATE TABLE IF NOT EXISTS `{{PROJECT}}.{{DATASET}}.calibration_metrics` (
  market STRING,
  horizon INT64,
  target STRING,
  as_of_date DATE,
  model STRING,
  brier FLOAT64,
  ece FLOAT64,
  n INT64,
  run_id STRING
)
PARTITION BY as_of_date
CLUSTER BY market, horizon, target, model;