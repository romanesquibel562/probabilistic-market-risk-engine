# Market Risk Forecasting Engine

An institutional-style, leakage-safe, multi-horizon probabilistic risk forecasting platform built in Python and BigQuery.

This system produces calibrated probability forecasts of downside market events across multiple time horizons (5d / 21d / 63d), along with calibration diagnostics, top-K alert evaluation, and walk-forward backtesting scaffolding.

---

## Overview

This project is designed to resemble a professional risk analytics engine rather than a simple predictive model.

It emphasizes:

- Strict anti-leakage data handling
- Versioned feature and target stores
- Multi-horizon event modeling
- Calibration discipline
- Diagnostic transparency
- Modular, production-oriented structure

The system is BigQuery-backed and built with extensibility toward multi-asset and portfolio-level risk aggregation.

---

## Current Capabilities

### Data Layer

- BigQuery warehouse integration
- Leakage-safe as-of reads
- Versioned raw, feature, and target tables
- Latest views enforcing uniqueness
- Idempotent daily pipeline

### Feature Engineering (v2)

Close-price feature family:

- Log returns (1d / 5d / 21d / 63d)
- Realized volatility (5d / 21d / 63d)
- Moving averages (MA20 / MA63)
- Price-to-MA ratios
- Drawdowns (21d / 63d)
- Volatility regime ratios

All features include:

- as_of_date
- available_time (anti-leakage enforcement)
- feature_version
- run_id
- computed_at

---

### Target Generation (v2)

Forward log return targets:

- fwd_ret_5d_log
- fwd_ret_21d_log
- fwd_ret_63d_log

Strict invariant enforced:

```
available_time = as_of_date + horizon_days (UTC midnight)
```

Targets are versioned and validated before modeling.

---

### Modeling (Step 6)

Multi-horizon logistic risk-event framework:

- Sigma-based downside event rule
- Logistic regression with class weighting
- Calibration candidates:
  - Raw
  - Platt (sigmoid)
  - Optional isotonic (short horizon only)
- Calibration guardrails:
  - Spread checks
  - p_std minimum threshold
  - Unique probability ratio checks
- Prior alignment with do-no-harm gate
- Rolling calibration drift diagnostics
- Reliability tables
- Top-K alert evaluation

Artifacts are automatically saved to:

```
artifacts/models/
```

---

## Architecture

```
Raw Ingest
    ↓
raw_series_values_v2 (BigQuery)
    ↓
Feature Builder (v2)
    ↓
features_v3 + features_latest_v3
    ↓
Target Builder (v2)
    ↓
targets_v3 + targets_latest_v3
    ↓
Training Matrix
    ↓
Multi-Horizon Event Models
    ↓
Calibration + Evaluation + Artifacts
```

---

## Setup

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd market_risk_engine
```

---

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Create Environment File

Copy template:

```bash
cp .env.example .env
```

Edit `.env`:

```
GCP_PROJECT_ID=your-project-id
BQ_DATASET=your_dataset
ENV=dev
```

Authenticate with GCP:

```bash
gcloud auth application-default login
```

---

## Running the Pipeline

### Full Daily Pipeline

```bash
python -m src.pipeline.daily_pipeline
```

---

### Train Multi-Horizon Event Models

```bash
python -m src.models.train_multi_horizon_events
```

---

### Walk-Forward Backtest

```bash
python -m src.models.backtest
```

---

## Example Output

For SPY (5d horizon):

- Calibrated Brier approximately 0.19
- AUC approximately 0.65
- Top 5 percent precision approximately 75 percent
- Rolling calibration diagnostics
- Reliability tables
- Saved model artifacts

---

## Design Principles

- No data leakage
- Strict versioning
- Multi-horizon modeling
- Calibration-first philosophy
- Transparent diagnostics
- Modular architecture
- Production-oriented structure

---

## Roadmap

Planned expansions:

- Walk-forward aggregate summary tables
- Additional event families:
  - Volatility expansion
  - Drawdown acceleration
  - Tail shock quantile events
- Expanded feature families:
  - Macro indicators
  - Credit spreads
  - Term structure features
  - Market breadth indicators
- Monte Carlo scenario engine
- Portfolio-level VaR and CVaR aggregation
- Dashboard interface
- Airflow DAG orchestration

---

## Status

Work in progress.

The ingestion, feature engineering, target generation, modeling, calibration, and backtesting systems are operational and validated.

Ongoing development focuses on expanding event families, enhancing evaluation harnesses, and preparing for multi-asset support.
