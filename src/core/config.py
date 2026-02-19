# src/core/config.py
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Load .env from proj root
load_dotenv(BASE_DIR / ".env")


class Settings:
    # Environment
    ENV = os.getenv("ENV", "dev")

    # GCP / BigQuery
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") or os.getenv("GCP_PROJECT")  # Support older env var name
    BQ_DATASET = os.getenv("BQ_DATASET")

    if not GCP_PROJECT_ID:
        raise ValueError("GCP_PROJECT_ID is not set in environment variables.")
    if not BQ_DATASET:
        raise ValueError("BQ_DATASET is not set in environment variables.")

    # -------------------------
    # Warehouse Versioning
    # -------------------------
    # These control what the training/matrix layer reads by default.
    # View names represent the "latest" consolidated views.
    FEATURES_LATEST_VIEW = os.getenv("FEATURES_LATEST_VIEW", "features_latest_v3")
    TARGETS_LATEST_VIEW = os.getenv("TARGETS_LATEST_VIEW", "targets_latest_v3")

    # These are the version tags stored in the rows inside the latest views.
    # Keep these aligned with what your pipeline currently writes.
    DEFAULT_FEATURE_VERSION = os.getenv("DEFAULT_FEATURE_VERSION", "v2")
    DEFAULT_TARGET_VERSION = os.getenv("DEFAULT_TARGET_VERSION", "v2")

    @staticmethod
    def load_yaml(relative_path: str):
        """
        Load a YAML config file from the project root.
        """
        path = BASE_DIR / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)


settings = Settings()
