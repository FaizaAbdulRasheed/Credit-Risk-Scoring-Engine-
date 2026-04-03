"""
Feature Engineering Pipeline
==============================
Creates engineered financial features and builds a sklearn
ColumnTransformer preprocessing pipeline.

Usage:
    python src/features/engineer.py --config config.yaml
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/features.log")


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering Functions
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features from raw columns.
    Each feature is derived from domain knowledge of credit risk.
    """
    df = df.copy()

    # 1. Payment-to-Income Ratio
    # Monthly payment as a fraction of monthly income — measures affordability
    df["payment_to_income"] = df["installment"] / (df["annual_inc"] / 12 + 1)

    # 2. Loan-to-Income Ratio
    # Total loan amount relative to annual income — signals over-borrowing
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)

    # 3. Credit Age in Months
    # Time since earliest credit line opened — longer history = lower risk
    if "earliest_cr_line" in df.columns and "issue_d" in df.columns:
        df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        df["credit_age_months"] = (
    (df["issue_d"] - df["earliest_cr_line"]).dt.days / 30.44
).clip(lower=0)
    else:
        df["credit_age_months"] = np.nan

    # 4. Log Income (normalise skewed distribution)
    df["income_log"] = np.log1p(df["annual_inc"].clip(lower=0))

    # 5. Delinquency Rate
    # Delinquencies normalised by number of open accounts
    df["delinq_rate"] = df["delinq_2yrs"] / (df["open_acc"] + 1)

    # 6. Employment Length (numeric, years)
    if "emp_length" in df.columns:
        df["emp_length_years_cat"] = df["emp_length"].map({
            "< 1 year": "0", "1 year": "1", "2 years": "2", "3 years": "3",
            "4 years": "4", "5 years": "5", "6 years": "6", "7 years": "7",
            "8 years": "8", "9 years": "9", "10+ years": "10+"
        }).fillna("Unknown")
    else:
        df["emp_length_years_cat"] = "Unknown"

    # 7. Clean revol_util (credit utilisation) — remove % sign if string
    if df["revol_util"].dtype == object:
        df["revol_util"] = pd.to_numeric(df["revol_util"].str.replace("%", ""), errors="coerce")

    # 8. Clean int_rate — remove % sign if string
    if "int_rate" in df.columns and df["int_rate"].dtype == object:
        df["int_rate"] = pd.to_numeric(df["int_rate"].str.replace("%", ""), errors="coerce")

    logger.info(f"Feature engineering complete | Shape: {df.shape}")
    return df


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that applies:
    - Median imputation + StandardScaling for numeric features
    - Mode imputation + OneHotEncoding for categorical features
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop"  # Drop any columns not specified
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """Get final feature names after preprocessing (including OHE columns)."""
    ohe = preprocessor.named_transformers_["cat"]["encoder"]
    cat_names = ohe.get_feature_names_out(categorical_features).tolist()
    return numeric_features + cat_names


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    Path("logs").mkdir(exist_ok=True)

    for split in ["train", "val", "test"]:
        path = config["paths"][f"{split}_data"]
        logger.info(f"Engineering features for {split} split...")
        df = pd.read_parquet(path)

        # Separate target
        y = df["target"]
        X = df.drop(columns=["target"])

        # Engineer features
        X = engineer_features(X)

        # Save back
        X["target"] = y.values
        X.to_parquet(path, index=False)
        logger.info(f"Saved engineered {split} split to {path}")

    # Build and fit preprocessor on training data only
    train_df = pd.read_parquet(config["paths"]["train_data"])
    X_train = train_df.drop(columns=["target"])

    numeric_features = [f for f in config["features"]["numeric"] if f in X_train.columns]
    categorical_features = [f for f in config["features"]["categorical"] if f in X_train.columns]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X_train)

    # Save preprocessor
    Path(config["paths"]["model_dir"]).mkdir(exist_ok=True)
    preprocessor_path = Path(config["paths"]["model_dir"]) / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")

    # Save feature names
    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)
    with open(config["paths"]["feature_names_path"], "w") as f:
        json.dump({"features": feature_names, "numeric": numeric_features, "categorical": categorical_features}, f, indent=2)
    logger.info(f"Feature names saved | Total features: {len(feature_names)}")

    logger.info("Feature engineering pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
