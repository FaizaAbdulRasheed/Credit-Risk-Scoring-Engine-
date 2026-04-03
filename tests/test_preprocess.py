"""
Unit Tests — Data Preprocessing Pipeline
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.preprocess import (
    create_binary_target,
    drop_high_null_columns,
    drop_leaky_columns,
)


MOCK_CONFIG = {
    "data": {
        "target_column": "loan_status",
        "positive_class_labels": ["Charged Off", "Default"],
        "negative_class_labels": ["Fully Paid"],
        "drop_statuses": ["Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"],
    },
    "project": {"random_seed": 42},
}


@pytest.fixture
def sample_loans():
    return pd.DataFrame({
        "loan_status": [
            "Fully Paid", "Charged Off", "Current",
            "Default", "In Grace Period", "Fully Paid", "Charged Off"
        ],
        "loan_amnt": [10000] * 7,
        "int_rate": [12.5] * 7,
    })


class TestCreateBinaryTarget:

    def test_drops_indeterminate_rows(self, sample_loans):
        df = create_binary_target(sample_loans, MOCK_CONFIG)
        # "Current" and "In Grace Period" should be dropped
        assert len(df) == 5

    def test_correct_default_labels(self, sample_loans):
        df = create_binary_target(sample_loans, MOCK_CONFIG)
        defaults = df[df["target"] == 1]["loan_status"].unique()
        assert set(defaults) <= {"Charged Off", "Default"}

    def test_correct_non_default_labels(self, sample_loans):
        df = create_binary_target(sample_loans, MOCK_CONFIG)
        non_defaults = df[df["target"] == 0]["loan_status"].unique()
        assert set(non_defaults) == {"Fully Paid"}

    def test_binary_target_only_0_1(self, sample_loans):
        df = create_binary_target(sample_loans, MOCK_CONFIG)
        assert set(df["target"].unique()) <= {0, 1}

    def test_default_rate_reasonable(self, sample_loans):
        df = create_binary_target(sample_loans, MOCK_CONFIG)
        # 2 Charged Off + 1 Default = 3 out of 5 = 60%
        assert 0.0 < df["target"].mean() < 1.0


class TestDropHighNullColumns:

    def test_drops_columns_exceeding_threshold(self):
        df = pd.DataFrame({
            "good_col": [1, 2, 3, 4, 5],
            "half_null": [1, None, 3, None, 5],
            "mostly_null": [None, None, None, None, 1],
        })
        result = drop_high_null_columns(df, max_null_pct=0.40)
        assert "good_col" in result.columns
        assert "mostly_null" not in result.columns

    def test_keeps_columns_below_threshold(self):
        df = pd.DataFrame({
            "col_a": [1, 2, 3, 4, 5],
            "col_b": [1, None, 3, 4, 5],  # 20% null
        })
        result = drop_high_null_columns(df, max_null_pct=0.40)
        assert "col_b" in result.columns

    def test_no_columns_dropped_when_all_clean(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = drop_high_null_columns(df, max_null_pct=0.40)
        assert list(result.columns) == ["a", "b"]


class TestDropLeakyColumns:

    def test_removes_known_leaky_columns(self):
        df = pd.DataFrame({
            "loan_amnt": [10000],
            "total_pymnt": [9500],        # leaky
            "recoveries": [0],             # leaky
            "last_pymnt_amnt": [350],      # leaky
            "int_rate": [12.5],
        })
        result = drop_leaky_columns(df)
        assert "total_pymnt" not in result.columns
        assert "recoveries" not in result.columns
        assert "last_pymnt_amnt" not in result.columns

    def test_keeps_non_leaky_columns(self):
        df = pd.DataFrame({
            "loan_amnt": [10000],
            "int_rate": [12.5],
            "annual_inc": [60000],
            "total_pymnt": [9500],  # leaky
        })
        result = drop_leaky_columns(df)
        assert "loan_amnt" in result.columns
        assert "int_rate" in result.columns
        assert "annual_inc" in result.columns

    def test_handles_missing_leaky_columns_gracefully(self):
        """Should not raise if leaky columns aren't present."""
        df = pd.DataFrame({"loan_amnt": [10000], "int_rate": [12.5]})
        result = drop_leaky_columns(df)
        assert len(result.columns) == 2
