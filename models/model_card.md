# Model Card — Credit Risk Scoring Engine v1.0

## Model Overview

| Field | Details |
|-------|---------|
| **Model Type** | XGBoost Binary Classifier (gradient boosted trees) |
| **Version** | 1.0.0 |
| **Task** | Loan default probability prediction |
| **Input** | 45 engineered features from loan application data |
| **Output** | Default probability ∈ [0, 1]; binary prediction at threshold 0.50 |
| **Framework** | Scikit-learn Pipeline + XGBoost 2.0.3 |

---

## Training Data

- **Dataset**: LendingClub Loan Data 2007–2018 (accepted loans)
- **Source**: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- **Size**: ~1.6M training samples after filtering
- **Target**: `loan_status` mapped to binary (Charged Off/Default = 1, Fully Paid = 0)
- **Class balance**: ~80% non-default, ~20% default
- **Temporal range**: January 2007 – Q4 2018

### Exclusions
- Loans with indeterminate outcomes (Current, Late, In Grace Period) were removed
- Loans recorded after the outcome was known (post-hoc features) were excluded to prevent data leakage

---

## Performance

Evaluated on held-out test set (20% stratified split, never seen during training or tuning):

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.88 |
| F1 Score | 0.83 |
| Precision | 0.85 |
| Recall | 0.81 |
| PR-AUC | 0.79 |

### Comparison with Baselines

| Model | ROC-AUC | F1 |
|-------|---------|-----|
| Logistic Regression | 0.72 | 0.69 |
| Random Forest | 0.83 | 0.77 |
| **XGBoost (this model)** | **0.88** | **0.83** |

---

## Hyperparameters (Best from GridSearchCV)

```yaml
max_depth: 5
n_estimators: 300
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
scale_pos_weight: 4
min_child_weight: 3
objective: binary:logistic
```

---

## Intended Use

- **Intended use**: Loan default risk assessment to support credit underwriting decisions
- **Intended users**: Credit risk analysts, underwriters, risk model validators
- **Out-of-scope**: Should NOT be used as the sole basis for credit decisions; human oversight required

---

## Limitations

1. **Temporal drift**: Economic conditions change. The model was trained on 2007–2018 data including the 2008 financial crisis. Performance may degrade in different economic regimes.
2. **Distribution shift**: Model assumes inference population resembles training population. Significant changes in applicant demographics or loan products require retraining.
3. **Protected attributes**: The model does not explicitly use race, gender, or national origin, but proxy discrimination via zip code or income may occur. Fairness audits are recommended before production deployment.
4. **Calibration**: While the model's probability estimates are directionally correct, they may not be perfectly calibrated. Use Platt scaling or isotonic regression for decision-making that requires calibrated probabilities.

---

## Fairness Considerations

Before production deployment:
- Run disparate impact analysis on protected classes (race, gender, national origin, age)
- Check for proxy discrimination via geographic features (addr_state)
- Apply IBM AI Fairness 360 or Aequitas for bias auditing
- Document results in compliance with ECOA and Fair Housing Act requirements

---

## Explainability

The model uses SHAP TreeExplainer for local (individual) and global (population-level) explanations.
Every prediction can be accompanied by a ranked list of features and their directional impact —
satisfying adverse action notice requirements under ECOA.

---

## Retraining Recommendations

- **Frequency**: Quarterly, or when ROC-AUC drops below 0.84 on monitoring data
- **Trigger**: PSI > 0.25 on key features (int_rate, dti, annual_inc)
- **Data**: Retrain on rolling 24-month window to balance recency and stability
