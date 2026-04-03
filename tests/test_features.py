"""
Unit Tests — Feature Engineering
Tests all engineered features to ensure correctness.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineer import engineer_features, build_preprocessor


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal valid input DataFrame."""
    return pd.DataFrame({
        "loan_amnt": [10000, 25000, 5000],
        "installment": [333.0, 500.0, 150.0],
        "annual_inc": [60000.0, 120000.0, 30000.0],
        "dti": [15.0, 25.0, 10.0],
        "delinq_2yrs": [0, 1, 2],
        "open_acc": [8, 12, 5],
        "revol_util": [55.0, 30.0, 80.0],
        "int_rate": [12.5, 18.0, 8.5],
        "emp_length": ["5 years", "10+ years", "< 1 year"],
        "earliest_cr_line": ["Jan-2010", "Mar-2005", "Jul-2015"],
        "issue_d": ["Jan-2020", "Jan-2020", "Jan-2020"],
        "fico_range_low": [700, 720, 680],
        "fico_range_high": [704, 724, 684],
    })


# ── Tests ──────────────────────────────────────────────────────────────────

class TestEngineerFeatures:

    def test_payment_to_income_positive(self, sample_df):
        df = engineer_features(sample_df)
        assert (df["payment_to_income"] >= 0).all()
        assert "payment_to_income" in df.columns

    def test_loan_to_income_positive(self, sample_df):
        df = engineer_features(sample_df)
        assert (df["loan_to_income"] >= 0).all()

    def test_loan_to_income_ratio(self, sample_df):
        """loan_to_income = loan_amnt / (annual_inc + 1)"""
        df = engineer_features(sample_df)
        expected = 10000 / (60000.0 + 1)
        assert abs(df.iloc[0]["loan_to_income"] - expected) < 1e-6

    def test_income_log_non_negative(self, sample_df):
        df = engineer_features(sample_df)
        assert (df["income_log"] >= 0).all()

    def test_income_log_formula(self, sample_df):
        df = engineer_features(sample_df)
        expected = np.log1p(60000.0)
        assert abs(df.iloc[0]["income_log"] - expected) < 1e-6

    def test_delinq_rate_non_negative(self, sample_df):
        df = engineer_features(sample_df)
        assert (df["delinq_rate"] >= 0).all()

    def test_delinq_rate_formula(self, sample_df):
        df = engineer_features(sample_df)
        expected = 0 / (8 + 1)  # delinq_2yrs=0, open_acc=8
        assert abs(df.iloc[0]["delinq_rate"] - expected) < 1e-6

    def test_credit_age_non_negative(self, sample_df):
        df = engineer_features(sample_df)
        assert (df["credit_age_months"].dropna() >= 0).all()

    def test_emp_length_mapping(self, sample_df):
        df = engineer_features(sample_df)
        assert "emp_length_years_cat" in df.columns
        assert df.iloc[2]["emp_length_years_cat"] == "0"  # < 1 year
        assert df.iloc[1]["emp_length_years_cat"] == "10+"  # 10+ years

    def test_no_inf_values(self, sample_df):
        df = engineer_features(sample_df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert not np.isinf(df[numeric_cols]).any().any()

    def test_zero_income_no_crash(self):
        """Edge case: zero income should not cause division by zero."""
        df = pd.DataFrame({
            "loan_amnt": [10000],
            "installment": [333.0],
            "annual_inc": [0.0],
            "dti": [0.0],
            "delinq_2yrs": [0],
            "open_acc": [5],
            "revol_util": [50.0],
            "int_rate": [12.5],
            "emp_length": ["5 years"],
        })
        result = engineer_features(df)
        assert not result["payment_to_income"].isna().any()
        assert not result["loan_to_income"].isna().any()

    def test_shape_preserved(self, sample_df):
        """Feature engineering should only ADD columns, not remove rows."""
        df = engineer_features(sample_df)
        assert len(df) == len(sample_df)


class TestBuildPreprocessor:

    def test_preprocessor_fits_and_transforms(self, sample_df):
        df = engineer_features(sample_df)
        numeric_features = ["loan_amnt", "annual_inc", "dti", "int_rate"]
        categorical_features = ["emp_length_years_cat"]

        preprocessor = build_preprocessor(numeric_features, categorical_features)
        X_transformed = preprocessor.fit_transform(df)

        assert X_transformed.shape[0] == len(df)
        assert X_transformed.shape[1] > len(numeric_features)  # OHE expands categorical

    def test_preprocessor_handles_nulls(self):
        """Preprocessor should handle NaN values via imputation."""
        df = pd.DataFrame({
            "loan_amnt": [10000, np.nan, 5000],
            "annual_inc": [60000, 80000, np.nan],
            "emp_length_years_cat": ["5", None, "3"],
        })
        preprocessor = build_preprocessor(["loan_amnt", "annual_inc"], ["emp_length_years_cat"])
        X_transformed = preprocessor.fit_transform(df)
        assert not np.isnan(X_transformed).any()
