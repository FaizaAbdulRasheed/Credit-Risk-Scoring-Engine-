import argparse, json, sys
from pathlib import Path
from typing import Dict, Tuple
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config
from src.utils.logger import get_logger
logger = get_logger(__name__, log_file="logs/train.log")

def load_all_splits(config):
    train_df = pd.read_parquet(config["paths"]["train_data"])
    val_df   = pd.read_parquet(config["paths"]["val_data"])
    test_df  = pd.read_parquet(config["paths"]["test_data"])
    y_train = train_df.pop("target")
    y_val   = val_df.pop("target")
    y_test  = test_df.pop("target")
    drop_cols = ["loan_status","id","member_id","url","desc","title",
                 "zip_code","earliest_cr_line","issue_d","last_credit_pull_d",
                 "last_pymnt_d","next_pymnt_d","emp_title","pymnt_plan",
                 "hardship_flag","debt_settlement_flag","disbursement_method"]
    for df in [train_df, val_df, test_df]:
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    numeric_cols     = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=["object","category"]).columns.tolist()
    logger.info(f"Numeric features: {len(numeric_cols)} | Categorical features: {len(categorical_cols)}")
    return train_df, val_df, test_df, y_train, y_val, y_test, numeric_cols, categorical_cols

def build_preprocessor(X_train, numeric_cols, categorical_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]),
         categorical_cols),
    ], remainder="drop")
    pre.fit(X_train)
    logger.info("Preprocessor fitted")
    return pre

def train_model(pre, X, y, params, seed):
    logger.info(f"Training XGBoost | samples={len(X):,} | params={params}")
    xgb = XGBClassifier(objective="binary:logistic", eval_metric="auc",
                        use_label_encoder=False, random_state=seed,
                        n_jobs=1, **params)
    pipe = Pipeline([("preprocessor", pre), ("classifier", xgb)])
    pipe.fit(X, y)
    logger.info("Training complete!")
    return pipe

def evaluate_model(pipe, X_test, y_test):
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    m = {"roc_auc": round(roc_auc_score(y_test, y_proba), 4),
         "f1_score": round(f1_score(y_test, y_pred), 4),
         "precision": round(precision_score(y_test, y_pred), 4),
         "recall": round(recall_score(y_test, y_pred), 4),
         "avg_precision": round(average_precision_score(y_test, y_proba), 4)}
    logger.info("="*50)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("="*50)
    for k,v in m.items(): logger.info(f"  {k:<20}: {v:.4f}")
    logger.info("="*50)
    logger.info("\n" + classification_report(y_test, y_pred,
                target_names=["Non-Default","Default"]))
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    return m

def main(config_path="config.yaml"):
    config = load_config(config_path)
    Path("logs").mkdir(exist_ok=True)
    seed = config["project"]["random_seed"]
    params = {k: v for k, v in config["model"]["params"].items()
              if k not in ["objective","eval_metric","use_label_encoder",
                           "random_state","n_jobs"]}
    (X_tr, X_va, X_te,
     y_tr, y_va, y_te,
     num_cols, cat_cols) = load_all_splits(config)
    logger.info(f"Train:{len(X_tr):,} Val:{len(X_va):,} Test:{len(X_te):,}")
    X_tv = pd.concat([X_tr, X_va]).reset_index(drop=True)
    y_tv = pd.concat([y_tr, y_va]).reset_index(drop=True)
    pre  = build_preprocessor(X_tv, num_cols, cat_cols)
    Path(config["paths"]["model_dir"]).mkdir(exist_ok=True)
    joblib.dump(pre, Path(config["paths"]["model_dir"]) / "preprocessor.joblib")
    with open(config["paths"]["feature_names_path"], "w") as f:
        json.dump({"numeric": num_cols, "categorical": cat_cols}, f, indent=2)
    pipe = train_model(pre, X_tv, y_tv, params, seed)
    m    = evaluate_model(pipe, X_te, y_te)
    joblib.dump(pipe, config["paths"]["model_path"])
    logger.info(f"Model saved to {config['paths']['model_path']}")
    with open(Path(config["paths"]["model_dir"]) / "results.json", "w") as f:
        json.dump({"metrics": m, "params": params}, f, indent=2)
    logger.info("Done!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    main(p.parse_args().config)
