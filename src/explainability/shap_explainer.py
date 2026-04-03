"""
Real-Time SHAP Explainer
=========================
Lightweight wrapper used by the Streamlit app to compute
SHAP values for a single loan prediction at inference time.
Initialised once and cached with @st.cache_resource.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LoanSHAPExplainer:
    """
    Wraps a trained pipeline and TreeExplainer for single-row inference.

    Example:
        explainer = LoanSHAPExplainer.from_artefacts("models/")
        result = explainer.explain(input_df)
        print(result["top_features"])
    """

    def __init__(self, pipeline, feature_names: List[str]):
        self.pipeline = pipeline
        self.feature_names = feature_names
        xgb_model = pipeline.named_steps["classifier"]
        self._explainer = shap.TreeExplainer(xgb_model)
        self._preprocessor = pipeline.named_steps["preprocessor"]

    @classmethod
    def from_artefacts(cls, model_dir: str = "models/", feature_names_path: str = "models/feature_names.json"):
        """Load from saved artefacts."""
        import json
        pipeline = joblib.load(Path(model_dir) / "xgboost_pipeline_v1.joblib")
        with open(feature_names_path) as f:
            info = json.load(f)
        # Full post-OHE feature names
        preprocessor = pipeline.named_steps["preprocessor"]
        try:
            ohe = preprocessor.named_transformers_["cat"]["encoder"]
            cat_names = list(ohe.get_feature_names_out(info["categorical"]))
        except Exception:
            cat_names = []
        numeric_names = [f for f in info["numeric"]]
        feature_names = numeric_names + cat_names
        return cls(pipeline, feature_names)

    def explain(self, X: pd.DataFrame) -> Dict:
        """
        Compute SHAP values and return top contributing features.

        Returns:
            dict with keys:
              - probability: float
              - shap_values: np.ndarray
              - top_features: list of (feature_name, shap_value) tuples
              - base_value: float (expected model output)
        """
        # Predict probability
        probability = float(self.pipeline.predict_proba(X)[0, 1])

        # Transform for SHAP
        X_transformed = self._preprocessor.transform(X)

        # SHAP values
        shap_vals = self._explainer.shap_values(X_transformed)[0]  # single row
        base_value = float(self._explainer.expected_value)

        # Build top features list
        feature_impact = list(zip(self.feature_names, shap_vals))
        feature_impact_sorted = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)

        return {
            "probability": probability,
            "shap_values": shap_vals,
            "base_value": base_value,
            "top_features": feature_impact_sorted[:15],
            "all_features": feature_impact,
        }

    def get_risk_tier(self, probability: float) -> Tuple[str, str]:
        """Return (tier_label, colour_hex) based on default probability."""
        if probability < 0.20:
            return "Low Risk", "#2E8B57"
        elif probability < 0.45:
            return "Medium Risk", "#E09A2B"
        elif probability < 0.65:
            return "High Risk", "#E05A2B"
        else:
            return "Very High Risk", "#C0392B"
