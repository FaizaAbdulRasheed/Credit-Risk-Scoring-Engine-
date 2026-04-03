"""
Data Preprocessing Pipeline
============================
Loads raw LendingClub CSV, maps target variable to binary,
removes indeterminate loan statuses, drops high-null columns,
and saves train/val/test Parquet splits.

Usage:
    python src/data_pipeline/preprocess.py --config config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/preprocess.log")


def load_raw_data(path: str, nrows: int = None) -> pd.DataFrame:
    """Load raw LendingClub CSV."""
    logger.info(f"Loading raw data from {path} ...")
    df = pd.read_csv(
    path,
    low_memory=False,
    nrows=nrows,
    compression="infer",
    dtype={
        "id": str,
        "member_id": str,
        "emp_title": str,
        "desc": str,
        "title": str,
        "zip_code": str,
        "addr_state": str,
        "earliest_cr_line": str,
        "last_credit_pull_d": str,
        "issue_d": str,
        "last_pymnt_d": str,
        "next_pymnt_d": str,
    }
)
# Drop the trailing notes rows LendingClub appends at the end
    df = df[df["loan_amnt"].notna()].copy()
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def create_binary_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Map loan_status to binary target:
      1 = Default (Charged Off, Default)
      0 = Non-default (Fully Paid)
    Drops indeterminate rows (Current, Late, etc.)
    """
    data_cfg = config["data"]
    target_col = data_cfg["target_column"]

    # Drop indeterminate statuses
    drop_mask = df[target_col].isin(data_cfg["drop_statuses"])
    logger.info(f"Dropping {drop_mask.sum():,} indeterminate rows")
    df = df[~drop_mask].copy()

    # Map to binary
    df["target"] = np.where(
        df[target_col].isin(data_cfg["positive_class_labels"]), 1, 0
    )

    default_rate = df["target"].mean()
    logger.info(
        f"Target created | Default rate: {default_rate:.2%} "
        f"({df['target'].sum():,} defaults / {len(df):,} total)"
    )
    return df


def drop_high_null_columns(df: pd.DataFrame, max_null_pct: float) -> pd.DataFrame:
    """Drop columns where null percentage exceeds max_null_pct."""
    null_pct = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > max_null_pct].index.tolist()
    logger.info(f"Dropping {len(cols_to_drop)} columns with >{max_null_pct:.0%} nulls")
    return df.drop(columns=cols_to_drop)


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are recorded AFTER loan outcome — data leakage.
    These would give perfect accuracy in training but fail in production.
    """
    leaky_cols = [
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
        "total_rec_late_fee", "recoveries", "collection_recovery_fee",
        "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
        "out_prncp", "out_prncp_inv", "loan_status",  # original target
    ]
    cols_to_drop = [c for c in leaky_cols if c in df.columns]
    logger.info(f"Dropping {len(cols_to_drop)} leaky columns")
    return df.drop(columns=cols_to_drop)


def split_data(df: pd.DataFrame, config: dict) -> tuple:
    """Stratified train/val/test split."""
    cfg = config["data"]
    seed = config["project"]["random_seed"]

    X = df.drop(columns=["target"])
    y = df["target"]

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        stratify=y,
        random_state=seed
    )

    # Second split: train vs val
    val_ratio = cfg["val_size"] / (1 - cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=seed
    )

    logger.info(
        f"Splits → Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, config: dict):
    """Save train/val/test splits as Parquet."""
    paths = config["paths"]
    Path("data/splits").mkdir(parents=True, exist_ok=True)

    for X, y, name in [(X_train, y_train, "train"), (X_val, y_val, "val"), (X_test, y_test, "test")]:
        df_out = X.copy()
        df_out["target"] = y.values
        out_path = paths[f"{name}_data"]
        df_out.to_parquet(out_path, index=False)
        logger.info(f"Saved {name} split to {out_path}")


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    Path("logs").mkdir(exist_ok=True)

    # Load
    nrows = config["data"].get("nrows", None)
    df = load_raw_data(config["paths"]["raw_data"], nrows=nrows)

    # Clean target
    df = create_binary_target(df, config)

    # Drop leaky columns
    df = drop_leaky_columns(df)

    # Drop high-null columns
    df = drop_high_null_columns(df, config["data"]["max_null_pct"])

    # Save processed
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(config["paths"]["processed_data"], index=False)
    logger.info(f"Saved processed data to {config['paths']['processed_data']}")

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, config)
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test, config)

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)