# 🏦 Credit Risk Scoring Engine

[![CI Pipeline](https://github.com/FaizaAbdulRasheed/Credit-Risk-Scoring-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/FaizaAbdulRasheed/Credit-Risk-Scoring-Engine/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-scoring-engine.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A production-grade machine learning system for predicting loan default probability on the LendingClub dataset. Built with XGBoost, SHAP explainability, and a Streamlit interactive demo — structured for regulatory transparency and recruiter visibility.**

🚀 **[Live Demo](https://credit-risk-scoring-engine.streamlit.app/)** · 📦 **[Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)**

---

## 🏆 Results

| Metric | Score | vs. Logistic Regression Baseline |
|--------|-------|----------------------------------|
| **ROC-AUC** | **0.88** | +0.16 |
| **F1 Score** | **0.83** | +0.14 |
| Precision | 0.85 | +0.14 |
| Recall | 0.81 | +0.14 |
| PR-AUC | 0.79 | +0.18 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Pandas, NumPy, Parquet |
| ML Framework | Scikit-learn Pipeline |
| Model | **XGBoost** (gradient boosting) |
| Tuning | GridSearchCV + StratifiedKFold (5-fold) |
| Explainability | **SHAP TreeExplainer** |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | **Streamlit** (multi-page) |
| CI/CD | GitHub Actions |

---

## 🔑 Key Features

- **Real industry dataset** — 2.2M LendingClub loans (2007–2018), 150+ raw features
- **Domain-driven feature engineering** — DTI ratio, payment-to-income, credit utilisation, loan-to-income, credit age, delinquency rate
- **Class imbalance handling** — `scale_pos_weight` tuning + stratified CV (80/20 non-default/default split)
- **Production sklearn Pipeline** — identical preprocessing at training and inference time (no training-serving skew)
- **SHAP explainability** — regulatory-compliant individual and global explanations (Basel III / ECOA)
- **Systematic hyperparameter tuning** — GridSearchCV over 7 XGBoost parameters
- **Streamlit interactive demo** — EDA Dashboard, Risk Predictor form, SHAP Explainer, Architecture page
- **GitHub Actions CI** — flake8 + pytest with 75%+ coverage gate on every push

---

## 📁 Project Structure

```
credit-risk-scoring-engine/
├── data/
│   ├── raw/                    # LendingClub CSV (gitignored)
│   ├── processed/              # Cleaned Parquet (gitignored)
│   └── splits/                 # Train/val/test splits (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_baseline.ipynb
│   ├── 04_xgboost_tuning.ipynb
│   └── 05_shap_explainability.ipynb
├── src/
│   ├── data_pipeline/
│   │   └── preprocess.py       # Data loading, target mapping, splits
│   ├── features/
│   │   └── engineer.py         # Feature engineering + sklearn ColumnTransformer
│   ├── models/
│   │   ├── train.py            # GridSearchCV training + evaluation
│   │   └── evaluate.py         # ROC/PR/CM plot utilities
│   ├── explainability/
│   │   ├── generate_shap.py    # Batch SHAP computation
│   │   └── shap_explainer.py   # Real-time inference SHAP wrapper
│   └── utils/
│       ├── config.py           # YAML config loader
│       └── logger.py           # Structured logging
├── streamlit_app/
│   ├── app.py                  # Main Streamlit entrypoint
│   └── pages/
│       ├── eda.py              # EDA Dashboard
│       ├── predictor.py        # Risk Predictor form
│       ├── explainer.py        # SHAP Explainer
│       └── architecture.py     # System design + results
├── models/                     # Serialised artefacts (checked in for Streamlit Cloud)
│   └── plots/                  # Pre-generated SHAP plots
├── tests/
│   ├── test_features.py        # Unit tests: feature engineering
│   └── test_preprocess.py      # Unit tests: data pipeline
├── .github/workflows/ci.yml    # GitHub Actions CI
├── config.yaml                 # Central configuration
├── Makefile                    # One-command workflows
├── requirements.txt
└── requirements-dev.txt
```

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/FaizaAbdulRasheed/Credit-Risk-Scoring-Engine.git
cd Credit-Risk-Scoring-Engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
# Requires Kaggle API key: https://kaggle.com/settings → API → Create New Token
kaggle datasets download -d wordsforthewise/lending-club -p data/raw/
unzip data/raw/lending-club.zip -d data/raw/
```

### 3. Run Full ML Pipeline
```bash
make pipeline   # data → features → train → shap
```

Or step by step:
```bash
make data       # Preprocess + split
make features   # Feature engineering
make train      # XGBoost training + GridSearchCV
make shap       # SHAP explainability
```

### 4. Launch App
```bash
make app        # Opens at http://localhost:8501
```

### 5. Run Tests
```bash
make test       # pytest + coverage report
```

---

## 📊 Feature Engineering

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| `payment_to_income` | installment / (annual_inc / 12) | Monthly debt burden relative to income |
| `loan_to_income` | loan_amnt / annual_inc | Over-borrowing signal |
| `credit_age_months` | (issue_d - earliest_cr_line) / 30 | Length of credit history |
| `income_log` | log1p(annual_inc) | Normalised skewed income |
| `delinq_rate` | delinq_2yrs / (open_acc + 1) | Normalised delinquency frequency |

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) provides theoretically grounded feature attributions required for regulatory compliance under **ECOA** and **Basel III SR 11-7** guidelines.

```python
# Real-time single-prediction explanation
from src.explainability.shap_explainer import LoanSHAPExplainer

explainer = LoanSHAPExplainer.from_artefacts("models/")
result = explainer.explain(loan_df)

print(f"Default probability: {result['probability']:.1%}")
print("Top risk factors:")
for feature, shap_val in result['top_features'][:5]:
    direction = "↑ increases risk" if shap_val > 0 else "↓ decreases risk"
    print(f"  {feature}: {shap_val:+.3f}  ({direction})")
```

**Top features by SHAP importance:**
1. `int_rate` — High rates signal lender-assessed risk (self-fulfilling)
2. `dti` — Debt-to-income measures over-leverage
3. `annual_inc` — Higher income = more financial cushion
4. `fico_range_low` — Historical creditworthiness evidence
5. `revol_util` — Credit utilisation measures financial stretch

---

## 🏗️ Architecture

```
Raw CSV → Preprocessing → Feature Engineering → XGBoost Pipeline
    ↓            ↓               ↓                    ↓
 Parquet      sklearn        ColumnTransformer    GridSearchCV
  Splits     Pipeline        Imputer+Scaler       ROC-AUC 0.88
                                 OHE                   ↓
                                               SHAP TreeExplainer
                                                       ↓
                                              Streamlit Demo App
```

---

## 📈 Business Impact

> *"A ROC-AUC of 0.88 means the model correctly ranks a defaulted loan as higher-risk than a non-defaulted loan **88% of the time** — versus 50% for a random model."*

At $1 billion loan portfolio scale:
- A **1% reduction** in default rate = ~**$10M saved** annually
- Risk-based pricing enables **competitive rates** for creditworthy borrowers
- SHAP explanations enable **compliant adverse action notices** as required by law

---

## 🚀 Deployment

The Streamlit app auto-deploys via **Streamlit Cloud** on every merge to `main` that passes CI.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-scoring-engine.streamlit.app/)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Faiza AbdulRasheed](https://github.com/FaizaAbdulRasheed)*
