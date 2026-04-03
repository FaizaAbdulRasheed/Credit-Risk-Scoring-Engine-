"""
Architecture & About Page
System design and project information.
"""
import streamlit as st


def show_architecture():
    st.title("🏗️ System Architecture")
    st.markdown("Production-grade ML system design — inspired by Uber Michelangelo and Google TFX patterns.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔄 Data Pipeline", "📊 Model Results", "👤 About"])

    with tab1:
        _pipeline_tab()

    with tab2:
        _results_tab()

    with tab3:
        _about_tab()


def _pipeline_tab():
    st.subheader("End-to-End ML Pipeline")

    # ASCII architecture diagram in a code block
    st.code("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    CREDIT RISK SCORING ENGINE                           │
│                     SYSTEM ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐
  │  DATA LAYER  │────▶│ FEATURE LAYER│────▶│      MODEL LAYER         │
  │              │     │              │     │                          │
  │ LendingClub  │     │ DTI Ratio    │     │ Logistic Regression ──┐  │
  │ CSV (2.2M)   │     │ Pay-to-Inc   │     │ Random Forest      ──┐│  │
  │              │     │ Loan-to-Inc  │     │ XGBoost (final) ◀──┘└┘  │
  │ Parquet      │     │ Credit Age   │     │                          │
  │ Splits       │     │ Income Log   │     │ GridSearchCV (5-fold)    │
  │ (70/10/20)   │     │ Delinq Rate  │     │ ROC-AUC 0.88, F1 0.83   │
  └──────────────┘     └──────────────┘     └──────────────────────────┘
          │                    │                          │
          │         ┌──────────┴──────────┐              │
          │         │  sklearn Pipeline   │              │
          │         │  ColumnTransformer  │              │
          │         │  Imputer + Scaler   │              │
          │         │  OHE Categorical    │              │
          │         └─────────────────────┘              │
          │                                              │
          ▼                                              ▼
  ┌──────────────────┐                      ┌─────────────────────────┐
  │ EXPLAINABILITY   │                      │   APPLICATION LAYER     │
  │                  │                      │                         │
  │ TreeExplainer    │                      │ Streamlit Multi-Page    │
  │ SHAP Values      │                      │ ├─ EDA Dashboard        │
  │ Summary Plot     │──────────────────────│ ├─ Risk Predictor       │
  │ Force Plots      │                      │ ├─ SHAP Explainer       │
  │ Dependence Plots │                      │ └─ Architecture/About   │
  └──────────────────┘                      └─────────────────────────┘
                                                         │
                                              ┌──────────┴──────────┐
                                              │     CI/CD LAYER     │
                                              │                     │
                                              │ GitHub Actions      │
                                              │ ├─ flake8 lint      │
                                              │ ├─ pytest (80%+cov) │
                                              │ └─ Streamlit Cloud  │
                                              └─────────────────────┘
    """, language=None)

    st.markdown("---")
    st.subheader("Key Design Decisions")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Why Parquet for data storage?**
        Parquet is a columnar format that's 5–10x faster than CSV for pandas operations
        and 3–5x smaller file size. Critical for 2.2M row datasets.

        **Why sklearn Pipeline?**
        Encapsulating all preprocessing ensures identical transforms at training
        and inference time — eliminates training-serving skew, the #1 source of
        production ML bugs.

        **Why StratifiedKFold?**
        With 80/20 class imbalance, random splits can produce folds with different
        default rates. Stratified CV preserves the ratio in each fold, giving
        reliable cross-validation scores.
        """)
    with col2:
        st.markdown("""
        **Why scale_pos_weight in XGBoost?**
        Directly addresses class imbalance by upweighting the positive (default)
        class. Set to ~4 (ratio of negatives to positives). Alternative approaches:
        SMOTE, class_weight, threshold tuning.

        **Why TreeExplainer over KernelExplainer?**
        TreeExplainer is exact (not approximate) for tree-based models. It runs
        in O(TLD²) time where T=trees, L=leaves, D=depth — fast enough for
        real-time single-row explanations in the Streamlit app.

        **Why save preprocessor separately?**
        Allows model updates without reprocessing data, and vice versa. The
        preprocessor is typically stable; the model is retrained quarterly.
        """)


def _results_tab():
    st.subheader("Model Performance Results")
    st.markdown("Evaluated on 20% held-out stratified test set (never seen during training or tuning).")

    import plotly.graph_objects as go

    # Model comparison table
    results = {
        "Logistic Regression": {"ROC-AUC": 0.72, "F1 Score": 0.69, "Precision": 0.71, "Recall": 0.67},
        "Random Forest": {"ROC-AUC": 0.83, "F1 Score": 0.77, "Precision": 0.79, "Recall": 0.75},
        "XGBoost (tuned)": {"ROC-AUC": 0.88, "F1 Score": 0.83, "Precision": 0.85, "Recall": 0.81},
    }

    import pandas as pd
    df_results = pd.DataFrame(results).T
    df_results = df_results.style.background_gradient(cmap="YlGn", vmin=0.6, vmax=0.95)
    st.dataframe(df_results, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### XGBoost Best Hyperparameters")
        params = {
            "max_depth": 5,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 4,
            "min_child_weight": 3,
        }
        for k, v in params.items():
            st.metric(k, v)

    with col2:
        st.markdown("### Business Impact Interpretation")
        st.markdown("""
        **ROC-AUC of 0.88** means:
        > For a randomly selected defaulted loan and a randomly selected
        > non-defaulted loan, the model ranks the defaulted loan as higher
        > risk **88% of the time**. A random model scores 0.50.

        **F1 Score of 0.83** means:
        > Harmonic mean of precision (85%) and recall (81%) — the model
        > catches 81% of actual defaults while maintaining 85% precision,
        > minimising both missed defaults and false alarms.

        **At $1B loan portfolio scale:**
        > A 1% reduction in default rate = ~$10M saved annually.
        > Model enables risk-based pricing that can also increase revenue
        > by correctly approving borderline-but-creditworthy applicants.
        """)


def _about_tab():
    st.subheader("About This Project")
    st.markdown("""
    ### Credit Risk Scoring Engine

    Built as a **FANG-level portfolio project** to demonstrate end-to-end
    production ML engineering skills in the fintech/credit risk domain.

    **Project Highlights:**
    - Real industry dataset (LendingClub, 2.2M loans)
    - Production sklearn Pipeline (no training-serving skew)
    - SHAP explainability (regulatory compliance)
    - Systematic hyperparameter tuning with GridSearchCV
    - GitHub Actions CI/CD pipeline
    - Clean separation of concerns: data / features / models / app

    **Technologies:**
    Python 3.11 · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM ·
    SHAP · Streamlit · Plotly · Joblib · Optuna · GitHub Actions

    ---

    ### Links
    - 🔗 **GitHub:** [github.com/FaizaAbdulRasheed/Credit-Risk-Scoring-Engine](https://github.com/FaizaAbdulRasheed/Credit-Risk-Scoring-Engine)
    - 🚀 **Live Demo:** [credit-risk-scoring-engine.streamlit.app](https://credit-risk-scoring-engine.streamlit.app/)
    - 📦 **Dataset:** [Kaggle — LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
    """)
