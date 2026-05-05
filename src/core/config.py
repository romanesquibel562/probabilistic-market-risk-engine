import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class Settings:
    def __init__(self) -> None:
        # Environment
        self.ENV = os.getenv("ENV", "dev")

        # GCP / BigQuery
        self.GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") or os.getenv("GCP_PROJECT")
        self.BQ_DATASET = os.getenv("BQ_DATASET")

        # Warehouse versioning
        self.FEATURES_LATEST_VIEW = os.getenv("FEATURES_LATEST_VIEW", "features_latest_v3")
        self.TARGETS_LATEST_VIEW = os.getenv("TARGETS_LATEST_VIEW", "targets_latest_v3")

        self.DEFAULT_FEATURE_VERSION = os.getenv("DEFAULT_FEATURE_VERSION", "v2")
        self.DEFAULT_TARGET_VERSION = os.getenv("DEFAULT_TARGET_VERSION", "v2")

    def validate_required(self) -> None:
        missing: list[str] = []

        if not self.GCP_PROJECT_ID:
            missing.append("GCP_PROJECT_ID")
        if not self.BQ_DATASET:
            missing.append("BQ_DATASET")

        if missing:
            raise ValueError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + f". Expected them in {BASE_DIR / '.env'} or the shell environment."
            )

    @staticmethod
    def load_yaml(relative_path: str):
        """
        Load a YAML config file from the project root.
        """
        path = BASE_DIR / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


settings = Settings()

