# src/training/training_runner.py
from __future__ import annotations

import datetime as dt

from src.training.training_matrix import build_training_matrix, split_train_test, SplitConfig


def main(test_days: int = 90) -> None:
    df = build_training_matrix(
        market="SPY",
        target_name="fwd_ret_5d_log",
        horizon_days=5,
        # feature_version / target_version intentionally omitted:
        # they come from settings.DEFAULT_FEATURE_VERSION / DEFAULT_TARGET_VERSION
    )

    if df.empty:
        raise RuntimeError(
            "Training matrix is empty. Check that your latest views exist and that "
            "settings.DEFAULT_FEATURE_VERSION / DEFAULT_TARGET_VERSION match the data."
        )

    # Use the dataframe's max date (not dt.date.today()) so the split is always valid
    max_d = dt.datetime.strptime(str(df["as_of_date"].max()), "%Y-%m-%d").date()
    min_d = dt.datetime.strptime(str(df["as_of_date"].min()), "%Y-%m-%d").date()

    test_end = max_d
    test_start = max_d - dt.timedelta(days=test_days) + dt.timedelta(days=1)
    if test_start < min_d:
        test_start = min_d

    train_end = test_start - dt.timedelta(days=1)

    split = SplitConfig(
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    train, test = split_train_test(df, split)

    print("Full rows:", len(df), "| date range:", df["as_of_date"].min(), "..", df["as_of_date"].max())
    print("Train rows:", len(train), "| date range:", train["as_of_date"].min(), "..", train["as_of_date"].max())
    print("Test rows:", len(test), "| date range:", test["as_of_date"].min(), "..", test["as_of_date"].max())

    print("\nColumns:", list(train.columns))
    print("\nTrain sample:")
    print(train.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

    # Run:
    #   python -m src.training.training_runner
