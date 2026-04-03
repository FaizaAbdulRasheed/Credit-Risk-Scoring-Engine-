"""
SHAP Explainer Page
Displays pre-generated SHAP plots and explains individual predictions.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

PLOTS_DIR = ROOT / "models" / "plots"


def show_explainer():
    st.title("🔍 SHAP Explainability")
    st.markdown(
        "**SHAP (SHapley Additive exPlanations)** provides theoretically grounded, "
        "model-agnostic explanations based on cooperative game theory. "
        "Required for Basel III / ECOA regulatory compliance in credit risk."
    )
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📊 Global Feature Importance",
        "⚡ How SHAP Works",
        "📖 Business Interpretation"
    ])

    with tab1:
        _global_importance_tab()

    with tab2:
        _how_shap_works_tab()

    with tab3:
        _business_interpretation_tab()


def _global_importance_tab():
    st.subheader("Global Feature Importance")
    st.markdown(
        "These plots are computed on 1,000 held-out test samples using `shap.TreeExplainer`. "
        "They show which features most influence the model's default predictions overall."
    )

    col1, col2 = st.columns(2)
    with col1:
        summary_path = PLOTS_DIR / "shap_summary.png"
        if summary_path.exists():
            st.markdown("#### Beeswarm Summary Plot")
            st.image(str(summary_path), use_column_width=True)
            st.caption(
                "Each dot is one loan. **Red = high feature value, Blue = low value**. "
                "Position on x-axis shows impact on default probability. "
                "Features ranked by mean |SHAP| value."
            )
        else:
            st.info("Run `make shap` to generate SHAP plots.")
            _show_demo_importance()

    with col2:
        bar_path = PLOTS_DIR / "shap_bar.png"
        if bar_path.exists():
            st.markdown("#### Top Features by Mean |SHAP| Value")
            st.image(str(bar_path), use_column_width=True)
            st.caption(
                "Mean absolute SHAP value across all test samples. "
                "Higher = more influential in predicting default."
            )
        else:
            _show_demo_importance_bar()


def _show_demo_importance():
    """Plotly demo when real SHAP plots aren't generated yet."""
    features = [
        "int_rate", "dti", "annual_inc", "fico_range_low", "revol_util",
        "loan_amnt", "payment_to_income", "grade", "delinq_2yrs", "open_acc"
    ]
    shap_mean = [0.41, 0.28, 0.24, 0.22, 0.19, 0.17, 0.15, 0.13, 0.11, 0.08]

    fig = go.Figure(go.Bar(
        x=shap_mean[::-1], y=features[::-1],
        orientation="h",
        marker_color="#2E5090"
    ))
    fig.update_layout(
        title="Top Features — Mean |SHAP| Value (Demo)",
        xaxis_title="Mean |SHAP Value|",
        height=380, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("📌 Demo values shown. Run `make shap` to generate real SHAP from your trained model.")


def _show_demo_importance_bar():
    features = [
        "int_rate", "dti", "annual_inc", "fico_range_low", "revol_util",
        "loan_amnt", "payment_to_income", "grade", "delinq_2yrs", "open_acc"
    ]
    shap_mean = [0.41, 0.28, 0.24, 0.22, 0.19, 0.17, 0.15, 0.13, 0.11, 0.08]
    colors = ["#E05A2B" if i < 3 else "#2E5090" for i in range(len(features))]

    fig = go.Figure(go.Bar(
        x=features, y=shap_mean,
        marker_color=colors, text=[f"{v:.2f}" for v in shap_mean],
        textposition="outside"
    ))
    fig.update_layout(
        title="Feature Importance (Demo)",
        yaxis_title="Mean |SHAP Value|",
        height=380, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


def _how_shap_works_tab():
    st.subheader("How SHAP Explains Predictions")
    st.markdown("""
    ### The Concept
    SHAP assigns each feature a "contribution" to moving a prediction away from the average prediction.
    It's based on **Shapley values** from cooperative game theory — the only method that satisfies
    four fairness axioms simultaneously (Efficiency, Symmetry, Dummy, Additivity).

    ### For Credit Risk
    For any loan prediction:
    ```
    Default Probability = Base Rate + SHAP(int_rate) + SHAP(dti) + SHAP(annual_inc) + ...
    ```

    - **Positive SHAP value** → feature pushes toward default
    - **Negative SHAP value** → feature pushes away from default
    - **Base rate** → model's average prediction on training data (~20% for LendingClub)

    ### Why Not Just Use Feature Importance?
    | Method | Local? | Consistent? | Handles Interactions? |
    |--------|--------|-------------|----------------------|
    | Permutation Importance | ❌ Global only | ✅ | ❌ |
    | LIME | ✅ | ❌ (approximation) | Partial |
    | SHAP | ✅ | ✅ | ✅ |

    SHAP TreeExplainer is **exact** (not approximate) for tree-based models like XGBoost.
    """)

    st.markdown("---")
    st.markdown("### Example Force Plot Interpretation")

    # Demo force plot using Plotly
    features_pos = [("int_rate = 24.5%", 0.18), ("dti = 38.2", 0.12), ("delinq_2yrs = 2", 0.08)]
    features_neg = [("annual_inc = $95k", -0.14), ("fico_score = 720", -0.11), ("grade = B", -0.07)]
    base = 0.18
    final = base + sum(v for _, v in features_pos) + sum(v for _, v in features_neg)

    fig = go.Figure()
    x_start = base
    for name, val in features_pos:
        fig.add_trace(go.Bar(
            x=[val], y=["Prediction"], orientation="h",
            base=x_start, name=name,
            marker_color="#E05A2B", showlegend=True,
            text=f"+{val:.2f}", textposition="inside"
        ))
        x_start += val
    for name, val in features_neg:
        fig.add_trace(go.Bar(
            x=[abs(val)], y=["Prediction"], orientation="h",
            base=x_start + val, name=name,
            marker_color="#2E5090", showlegend=True,
            text=f"{val:.2f}", textposition="inside"
        ))
        x_start += val

    fig.add_vline(x=base, line_dash="dash", line_color="gray",
                  annotation_text=f"Base rate: {base:.0%}")
    fig.add_vline(x=final, line_color="black", line_width=2,
                  annotation_text=f"Final: {final:.0%}")

    fig.update_layout(
        barmode="stack", height=180, template="plotly_white",
        title="SHAP Force Plot — Decomposing a Single Prediction",
        xaxis=dict(tickformat=".0%", title="Default Probability"),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔴 Red features INCREASE default probability | 🔵 Blue features DECREASE it. "
        "Final prediction = base rate + sum of all SHAP contributions."
    )


def _business_interpretation_tab():
    st.subheader("Business & Regulatory Interpretation")
    st.markdown("""
    ### Top Features & What They Mean

    | Feature | Direction | Business Interpretation |
    |---------|-----------|------------------------|
    | `int_rate` | ↑ Higher = More default | High interest rate → borrower is deemed riskier by underwriters → confirms signal |
    | `dti` | ↑ Higher = More default | High debt-to-income → over-leveraged borrower → less capacity to absorb shocks |
    | `annual_inc` | ↑ Higher = Less default | Higher income → more financial cushion to weather hardship |
    | `fico_range_low` | ↑ Higher = Less default | Better credit score → historical evidence of responsible repayment |
    | `revol_util` | ↑ Higher = More default | High credit utilisation → borrower is stretched across multiple lines |
    | `payment_to_income` | ↑ Higher = More default | High monthly burden → single income shock causes default |
    | `delinq_2yrs` | ↑ Higher = More default | Past delinquency → strongest predictor of future default behaviour |
    | `loan_amnt` | ↑ Higher = More default | Larger loan → more exposure, harder to fully repay |

    ---

    ### Regulatory Compliance
    Under **ECOA (Equal Credit Opportunity Act)**, lenders must be able to provide
    **specific reasons** for adverse credit actions. SHAP enables this:

    > *"Your application was declined primarily because: (1) your debt-to-income ratio of 38.2% exceeds
    > our guideline of 30%, (2) your interest rate tier indicates elevated risk, and (3) you have 2
    > delinquencies in the past 24 months."*

    This maps directly to the top SHAP contributors for that individual loan — a significant advantage
    over black-box models that can't produce interpretable adverse action notices.

    ---

    ### Fairness Considerations
    Before production deployment, fairness audits should be conducted on protected attributes
    (race, gender, national origin) that may be proxied by zip code or income. Tools:
    - **IBM AI Fairness 360** — disparate impact testing
    - **Aequitas** — bias auditing toolkit
    - **SHAP interaction values** — identify proxy discrimination
    """)
