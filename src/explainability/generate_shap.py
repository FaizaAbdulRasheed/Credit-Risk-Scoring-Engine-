"""
SHAP Explainability Generator
================================
Computes SHAP values and generates summary/bar/dependence plots.

Usage:
    python src/explainability/generate_shap.py --config config.yaml
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/shap.log")
PLOTS_DIR = Path("models/plots")


def get_feature_names_from_preprocessor(preprocessor) -> list:
    """Extract the exact feature names the preprocessor outputs."""
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "steps"):
            # It's a Pipeline — get the last step
            last_step = transformer.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(cols).tolist()
            else:
                names = list(cols)
        elif hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(cols).tolist()
        else:
            names = list(cols)
        feature_names.extend(names)

    return feature_names


def load_test_sample(config: dict, sample_size: int):
    """Load a random sample from the test set."""
    df = pd.read_parquet(config["paths"]["test_data"])
    y  = df.pop("target")

    # Drop non-feature columns (same as train.py)
    drop_cols = ["loan_status", "id", "member_id", "url", "desc", "title",
                 "zip_code", "earliest_cr_line", "issue_d", "last_credit_pull_d",
                 "last_pymnt_d", "next_pymnt_d", "emp_title", "pymnt_plan",
                 "hardship_flag", "debt_settlement_flag", "disbursement_method"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    seed   = config["project"]["random_seed"]
    sample = df.sample(min(sample_size, len(df)), random_state=seed)
    return sample, y.loc[sample.index]


def compute_shap_values(pipeline, X_sample: pd.DataFrame):
    """Transform data and compute SHAP TreeExplainer values."""
    logger.info(f"Computing SHAP values for {len(X_sample):,} samples...")

    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model    = pipeline.named_steps["classifier"]

    # Transform
    X_transformed = preprocessor.transform(X_sample)

    # Get exact feature names from preprocessor
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    logger.info(f"Transformed shape: {X_transformed.shape} | Feature names: {len(feature_names)}")

    # Align if there's any mismatch (safety net)
    n_cols = X_transformed.shape[1]
    if len(feature_names) != n_cols:
        logger.warning(
            f"Feature name count mismatch: {len(feature_names)} names vs "
            f"{n_cols} columns. Using generic names."
        )
        feature_names = [f"feature_{i}" for i in range(n_cols)]

    # Compute SHAP
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)

    logger.info(f"SHAP values computed | Shape: {shap_values.shape}")
    return shap_values, X_transformed, explainer, feature_names


def generate_summary_plot(shap_values, X_transformed, feature_names):
    logger.info("Generating SHAP summary beeswarm plot...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame so SHAP uses column names directly
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, max_display=20, show=False, plot_type="dot")
    plt.title("SHAP Summary — Feature Impact on Default Probability",
              fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Summary plot saved to {PLOTS_DIR / 'shap_summary.png'}")


def generate_bar_plot(shap_values, feature_names):
    logger.info("Generating SHAP bar plot...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, feature_names=feature_names,
                      max_display=20, show=False, plot_type="bar")
    plt.title("Top 20 Features by Mean |SHAP| Value",
              fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Bar plot saved to {PLOTS_DIR / 'shap_bar.png'}")


def generate_dependence_plots(shap_values, X_transformed, feature_names):
    logger.info("Generating SHAP dependence plots for top 3 features...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    mean_abs   = np.abs(shap_values).mean(axis=0)
    top3_idx   = np.argsort(mean_abs)[::-1][:3]
    X_df       = pd.DataFrame(X_transformed, columns=feature_names)

    for idx in top3_idx:
        feat_name  = feature_names[idx]
        safe_name  = feat_name.replace("/", "_").replace(" ", "_")[:50]
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(idx, shap_values, X_df, show=False)
        plt.title(f"SHAP Dependence — {feat_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"shap_dep_{safe_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
    logger.info(f"Dependence plots saved to {PLOTS_DIR}")


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    Path("logs").mkdir(exist_ok=True)

    # Load model
    pipeline = joblib.load(config["paths"]["model_path"])
    logger.info(f"Model loaded from {config['paths']['model_path']}")

    # Load test sample
    X_sample, y_sample = load_test_sample(config, config["shap"]["sample_size"])

    # Compute SHAP
    shap_values, X_transformed, explainer, feature_names = compute_shap_values(
        pipeline, X_sample
    )

    # Save artefacts
    np.save(config["paths"]["shap_values_path"], shap_values)
    X_sample.to_parquet(config["paths"]["shap_sample_path"], index=False)
    joblib.dump(explainer, Path(config["paths"]["model_dir"]) / "shap_explainer.joblib")

    # Save feature names for Streamlit app
    with open(config["paths"]["feature_names_path"], "w") as f:
        json.dump({"features": feature_names}, f, indent=2)

    logger.info("SHAP artefacts saved")

    # Generate plots
    generate_summary_plot(shap_values, X_transformed, feature_names)
    generate_bar_plot(shap_values, feature_names)
    generate_dependence_plots(shap_values, X_transformed, feature_names)

    logger.info("SHAP explainability generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)