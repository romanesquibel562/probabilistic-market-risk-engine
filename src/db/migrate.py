# src/db/migrate.py
from pathlib import Path
from google.cloud import bigquery
from src.core.config import settings


def run_migrations():
    client = bigquery.Client(project=settings.GCP_PROJECT_ID)

    sql_path = Path(__file__).parent / "schema.sql"
    sql = sql_path.read_text()

    sql = sql.replace("{{PROJECT}}", settings.GCP_PROJECT_ID)
    sql = sql.replace("{{DATASET}}", settings.BQ_DATASET)

    client.query(sql).result()

    print("BigQuery schema migration completed successfully.")


if __name__ == "__main__":
    run_migrations()

    # python -m src.db.migrate