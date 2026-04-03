"""
Model Evaluation Utilities
============================
Standalone evaluation helpers used by training script
and notebooks. Generates ROC curve, PR curve, and
confusion matrix plots saved to models/plots/.
"""
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_roc_curve(y_true, y_proba, save_path: str = None):
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2E5090", lw=2, label=f"XGBoost (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2E5090")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Credit Risk Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    return fig


def plot_pr_curve(y_true, y_proba, save_path: str = None):
    """Plot Precision-Recall curve with AP annotation."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = auc(recall, precision)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#E05A2B", lw=2, label=f"XGBoost (AP = {ap:.4f})")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline (AP = {baseline:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Credit Risk Model", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"PR curve saved to {save_path}")

    return fig


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """Plot normalised confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Non-Default", "Default"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
        normalize="true"
    )
    ax.set_title("Confusion Matrix (Normalised)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def compare_models(results: Dict[str, Dict], save_path: str = None):
    """
    Bar chart comparing multiple models on key metrics.

    Args:
        results: Dict of {model_name: {metric: value}}
        e.g. {"Logistic Regression": {"roc_auc": 0.72, "f1_score": 0.65}, ...}
    """
    metrics = ["roc_auc", "f1_score", "precision", "recall"]
    metric_labels = ["ROC-AUC", "F1 Score", "Precision", "Recall"]
    models = list(results.keys())
    colors = ["#2E5090", "#E05A2B", "#2E8B57", "#8B2E5A"]

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [results[model].get(m, 0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Model comparison chart saved to {save_path}")

    return fig
