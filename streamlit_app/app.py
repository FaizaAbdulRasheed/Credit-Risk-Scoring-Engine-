"""
Credit Risk Scoring Engine — Streamlit App
Main entrypoint. Run: streamlit run streamlit_app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Credit Risk Scoring Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    
    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F1E3C !important;
    }

    /* All sidebar text white */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] a {
        color: #E8EDF5 !important;
    }

    /* Sidebar metric cards */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: #1A3060 !important;
        border-left: 3px solid #4A90D9 !important;
        border-radius: 6px;
        padding: 8px 12px;
    }

    /* Metric label */
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
        color: #A8C0E8 !important;
        font-size: 12px !important;
    }

    /* Metric value */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 22px !important;
        font-weight: bold !important;
    }

    /* Metric delta */
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        color: #5CDB95 !important;
    }

    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio label {
        color: #E8EDF5 !important;
        font-size: 15px !important;
    }

    /* Main page headers */
    h1 { color: #1F3864; }
    h2 { color: #2E5090; }
    h3 { color: #375A8C; }

    /* Main metric cards */
    .main [data-testid="stMetric"] {
        background: linear-gradient(135deg, #EBF0FA, #F5F7FC);
        border-left: 4px solid #2E5090;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#FFFFFF;margin-bottom:4px;'>🏦 Credit Risk Engine</h2>",
            unsafe_allow_html=True
        )
        st.markdown("<hr style='border-color:#2A4A80;margin:8px 0;'>", unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            options=[
                "🏠 Overview",
                "📊 EDA Dashboard",
                "🔮 Risk Predictor",
                "🔍 SHAP Explainer",
                "🏗️ Architecture",
            ],
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border-color:#2A4A80;margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#A8C0E8;font-size:13px;font-weight:600;"
            "text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>"
            "Model Performance</p>",
            unsafe_allow_html=True
        )
        st.metric("ROC-AUC", "0.956", delta="+0.24 vs baseline")
        st.metric("F1 Score", "0.769", delta="+0.10 vs baseline")

        st.markdown("<hr style='border-color:#2A4A80;margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-Source-white?logo=github&style=flat)](https://github.com/FaizaAbdulRasheed/Credit-Risk-Scoring-Engine-)",
        )
        st.markdown(
            "[![Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=flat)](https://credit-risk-scoring-engine.streamlit.app/)",
        )

    # Route pages
    if page == "🏠 Overview":
        show_overview()
    elif page == "📊 EDA Dashboard":
        from streamlit_app.pages.eda import show_eda
        show_eda()
    elif page == "🔮 Risk Predictor":
        from streamlit_app.pages.predictor import show_predictor
        show_predictor()
    elif page == "🔍 SHAP Explainer":
        from streamlit_app.pages.explainer import show_explainer
        show_explainer()
    elif page == "🏗️ Architecture":
        from streamlit_app.pages.architecture import show_architecture
        show_architecture()


def show_overview():
    st.title("🏦 Credit Risk Scoring Engine")
    st.markdown(
        "**A production-grade machine learning system for loan default prediction, "
        "built with XGBoost + SHAP explainability for regulatory transparency.**"
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC", "0.956", delta="+0.24 vs LR baseline")
    with col2:
        st.metric("F1 Score", "0.769", delta="+0.10 vs LR baseline")
    with col3:
        st.metric("Training Samples", "313K", delta="LendingClub 2007–2018")
    with col4:
        st.metric("Features", "206", delta="after OHE encoding")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🎯 Problem Statement")
        st.markdown("""
        Predict the probability that a LendingClub personal loan will result
        in a **charge-off or default**, enabling lenders to:
        - Set **risk-based interest rates** appropriately
        - Make **data-driven approval decisions**
        - Provide **regulatory-compliant explanations** (ECOA, Basel III)
        """)

    with col_r:
        st.subheader("🛠️ Tech Stack")
        st.markdown("""
        | Layer | Technology |
        |-------|-----------|
        | Data | Pandas, NumPy, Parquet |
        | ML | Scikit-learn Pipeline |
        | Model | XGBoost |
        | Explainability | SHAP TreeExplainer |
        | Visualisation | Plotly, Matplotlib |
        | App | Streamlit |
        | CI/CD | GitHub Actions |
        """)

    st.markdown("---")
    st.subheader("🏆 Key Achievements")
    st.markdown("""
    - **ROC-AUC 0.956** — exceeds target of 0.88; model ranks defaulted loans correctly 95.6% of the time
    - **SHAP Explainability** — every prediction includes a feature-level breakdown for regulatory compliance
    - **Production sklearn Pipeline** — identical preprocessing at training and inference time
    - **Class Imbalance Handling** — scale_pos_weight=4 addresses the 80/20 non-default/default split
    - **206 features** — engineered from 75 raw columns including DTI, payment-to-income, credit utilisation
    """)

    st.markdown("---")
    st.info(
        "👈 **Use the sidebar to navigate** — try the **Risk Predictor** to score a loan, "
        "or **SHAP Explainer** to understand what drives default risk."
    )


if __name__ == "__main__":
    main()