"""
Risk Predictor Page
Interactive form to score a single loan application.
Fills all 75+ model features with sensible defaults.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def show_predictor():
    st.title("🔮 Loan Risk Predictor")
    st.markdown(
        "Enter loan application details. The model returns a **default probability** "
        "and **risk tier** using the trained XGBoost pipeline."
    )
    st.markdown("---")

    # ── Input Form ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("💳 Loan Details")
        loan_amnt   = st.slider("Loan Amount ($)", 1000, 40000, 12000, step=500)
        term        = st.selectbox("Term", ["36 months", "60 months"])
        purpose     = st.selectbox("Purpose", [
            "debt_consolidation", "credit_card", "home_improvement",
            "small_business", "major_purchase", "medical", "car", "other"
        ])
        int_rate    = st.slider("Interest Rate (%)", 5.0, 31.0, 13.5, step=0.25)
        grade       = st.selectbox("Loan Grade", list("ABCDEFG"), index=2)

    with col2:
        st.subheader("👤 Borrower Profile")
        annual_inc     = st.number_input("Annual Income ($)", 10000, 500000, 65000, step=5000)
        emp_length     = st.selectbox("Employment Length", [
            "< 1 year", "1 year", "2 years", "3 years", "4 years",
            "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
        ])
        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        addr_state     = st.selectbox("State", [
            "CA","TX","NY","FL","IL","PA","OH","GA","NC","MI",
            "NJ","VA","WA","AZ","MA","TN","IN","MO","MD","WI","Other"
        ])

    with col3:
        st.subheader("📋 Credit History")
        fico_range_low = st.slider("FICO Score", 600, 850, 700, step=5)
        dti            = st.slider("Debt-to-Income (%)", 0.0, 50.0, 18.0, step=0.5)
        revol_util     = st.slider("Credit Utilization (%)", 0.0, 100.0, 55.0, step=1.0)
        delinq_2yrs    = st.selectbox("Delinquencies (2yr)", [0, 1, 2, 3, 4, 5])
        open_acc       = st.slider("Open Credit Lines", 1, 40, 11)
        pub_rec        = st.selectbox("Public Records", [0, 1, 2, 3])

    st.markdown("---")

    if st.button("🚀 Calculate Risk Score", type="primary", use_container_width=True):
        with st.spinner("Running model inference..."):
            installment = _estimate_installment(loan_amnt, int_rate, term)
            input_df    = _build_full_input(
                loan_amnt, term, purpose, int_rate, grade,
                annual_inc, emp_length, home_ownership, addr_state,
                fico_range_low, dti, revol_util, delinq_2yrs,
                open_acc, pub_rec, installment
            )
            probability, used_model = _get_prediction(input_df)

        _display_results(probability, installment, annual_inc,
                         dti, int_rate, grade, used_model)


def _estimate_installment(loan_amnt, int_rate, term):
    months = 36 if "36" in term else 60
    r = (int_rate / 100) / 12
    if r == 0:
        return loan_amnt / months
    return loan_amnt * r * (1 + r)**months / ((1 + r)**months - 1)


def _build_full_input(loan_amnt, term, purpose, int_rate, grade,
                      annual_inc, emp_length, home_ownership, addr_state,
                      fico_range_low, dti, revol_util, delinq_2yrs,
                      open_acc, pub_rec, installment) -> pd.DataFrame:
    """
    Build a single-row DataFrame with ALL columns the model was trained on.
    Missing fields are filled with population medians / sensible defaults.
    """
    revol_bal      = annual_inc * 0.15
    total_acc      = open_acc + 8
    tot_cur_bal    = annual_inc * 1.2
    avg_cur_bal    = tot_cur_bal / max(open_acc, 1)

    row = {
        # ── Core loan fields ──────────────────────────────────────────
        "loan_amnt":         loan_amnt,
        "funded_amnt":       loan_amnt,
        "funded_amnt_inv":   loan_amnt,
        "term":              term,
        "int_rate":          int_rate,
        "installment":       installment,
        "grade":             grade,
        "sub_grade":         f"{grade}3",
        "emp_length":        emp_length,
        "home_ownership":    home_ownership,
        "annual_inc":        annual_inc,
        "verification_status": "Not Verified",
        "purpose":           purpose,
        "addr_state":        addr_state,
        "dti":               dti,
        "application_type":  "Individual",
        "initial_list_status": "w",

        # ── Credit history ────────────────────────────────────────────
        "delinq_2yrs":       delinq_2yrs,
        "fico_range_low":    fico_range_low,
        "fico_range_high":   fico_range_low + 4,
        "last_fico_range_low":  fico_range_low - 5,
        "last_fico_range_high": fico_range_low - 1,
        "inq_last_6mths":    1,
        "open_acc":          open_acc,
        "pub_rec":           pub_rec,
        "revol_bal":         revol_bal,
        "revol_util":        revol_util,
        "total_acc":         total_acc,

        # ── Balance / utilisation fields ──────────────────────────────
        "tot_cur_bal":       tot_cur_bal,
        "avg_cur_bal":       avg_cur_bal,
        "tot_coll_amt":      0,
        "total_rev_hi_lim":  revol_bal / max(revol_util / 100, 0.01),
        "tot_hi_cred_lim":   tot_cur_bal * 1.3,
        "total_bal_ex_mort": tot_cur_bal * 0.6,
        "total_il_high_credit_limit": tot_cur_bal * 0.4,
        "total_bc_limit":    revol_bal * 1.5,
        "bc_open_to_buy":    revol_bal * 0.3,
        "bc_util":           revol_util,

        # ── Account history counts ─────────────────────────────────────
        "acc_open_past_24mths":   3,
        "acc_now_delinq":         0,
        "delinq_amnt":            0,
        "num_accts_ever_120_pd":  0,
        "num_actv_bc_tl":         max(open_acc // 3, 1),
        "num_actv_rev_tl":        max(open_acc // 2, 1),
        "num_bc_sats":            max(open_acc // 3, 1),
        "num_bc_tl":              max(open_acc // 3, 2),
        "num_il_tl":              max(open_acc // 4, 1),
        "num_op_rev_tl":          max(open_acc // 2, 1),
        "num_rev_accts":          max(open_acc // 2, 2),
        "num_rev_tl_bal_gt_0":    max(open_acc // 3, 1),
        "num_sats":               open_acc,
        "num_tl_120dpd_2m":       0,
        "num_tl_30dpd":           0,
        "num_tl_90g_dpd_24m":     delinq_2yrs,
        "num_tl_op_past_12m":     2,
        "pct_tl_nvr_dlq":         max(95.0 - delinq_2yrs * 10, 60.0),
        "percent_bc_gt_75":       revol_util * 0.8,
        "mort_acc":               1 if home_ownership == "MORTGAGE" else 0,
        "pub_rec_bankruptcies":   0,
        "tax_liens":              0,
        "chargeoff_within_12_mths": 0,
        "collections_12_mths_ex_med": 0,
        "policy_code":            1,

        # ── Time-based features ───────────────────────────────────────
        "mo_sin_old_il_acct":     60,
        "mo_sin_old_rev_tl_op":   84,
        "mo_sin_rcnt_rev_tl_op":  6,
        "mo_sin_rcnt_tl":         4,
        "mths_since_recent_bc":   8,
        "mths_since_recent_inq":  3,

        # ── Engineered features ───────────────────────────────────────
        "payment_to_income":  installment / (annual_inc / 12 + 1),
        "loan_to_income":     loan_amnt / (annual_inc + 1),
        "credit_age_months":  120.0,
        "income_log":         np.log1p(annual_inc),
        "delinq_rate":        delinq_2yrs / (open_acc + 1),
        "emp_length_years_cat": _map_emp_length(emp_length),
    }

    return pd.DataFrame([row])


def _map_emp_length(emp_length: str) -> str:
    mapping = {
        "< 1 year": "0", "1 year": "1", "2 years": "2",
        "3 years": "3",  "4 years": "4", "5 years": "5",
        "6 years": "6",  "7 years": "7", "8 years": "8",
        "9 years": "9",  "10+ years": "10+"
    }
    return mapping.get(emp_length, "5")


def _get_prediction(input_df: pd.DataFrame):
    """Try real model; fallback to rule-based demo scorer."""
    model_path = ROOT / "models" / "xgboost_pipeline_v1.joblib"
    if model_path.exists():
        try:
            import joblib
            pipeline = joblib.load(model_path)
            prob = float(pipeline.predict_proba(input_df)[0, 1])
            return prob, "XGBoost (trained model)"
        except Exception as e:
            st.warning(f"Model error: {e}. Using demo scorer.")

    # Rule-based fallback
    row   = input_df.iloc[0]
    score = (
        -3.5
        + 0.12 * row["int_rate"]
        + 0.04 * row["dti"]
        - 0.006 * (row["fico_range_low"] - 670)
        + 0.08 * row["revol_util"] / 10
        + 0.25 * row["delinq_2yrs"]
        + {"A": -1.2, "B": -0.6, "C": 0.0,
           "D": 0.5, "E": 1.0, "F": 1.5, "G": 2.0}.get(row["grade"], 0)
    )
    return float(1 / (1 + np.exp(-score))), "Demo scorer"


def _get_risk_tier(prob):
    if prob < 0.20:   return "🟢 Low Risk",       "#2E8B57"
    elif prob < 0.45: return "🟡 Medium Risk",     "#E09A2B"
    elif prob < 0.65: return "🔴 High Risk",       "#E05A2B"
    else:             return "⛔ Very High Risk",  "#C0392B"


def _display_results(probability, installment, annual_inc,
                     dti, int_rate, grade, used_model):
    tier_label, tier_color = _get_risk_tier(probability)

    st.markdown("## 📊 Risk Assessment Result")
    col_score, col_gauge = st.columns([1, 2])

    with col_score:
        st.markdown(f"""
        <div style='text-align:center;padding:20px;
                    background:linear-gradient(135deg,#EBF0FA,#F5F7FC);
                    border-radius:12px;border-left:5px solid {tier_color};'>
            <p style='font-size:14px;color:#666;margin:0;'>Default Probability</p>
            <h1 style='font-size:52px;color:{tier_color};margin:4px 0;'>
                {probability:.1%}</h1>
            <div style='display:inline-block;padding:6px 18px;border-radius:20px;
                        background:{tier_color};color:white;font-weight:bold;
                        font-size:16px;'>{tier_label}</div>
            <p style='font-size:12px;color:#888;margin-top:10px;'>{used_model}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### Monthly Payment")
        st.metric("Estimated Installment", f"${installment:,.2f}/mo")
        pti = (installment / (annual_inc / 12)) * 100
        st.metric("Payment-to-Income",     f"{pti:.1f}%")

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 36}},
            title={"text": "Default Probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": tier_color, "thickness": 0.3},
                "steps": [
                    {"range": [0,  20], "color": "#D5F5E3"},
                    {"range": [20, 45], "color": "#FCF3CF"},
                    {"range": [45, 65], "color": "#FDEBD0"},
                    {"range": [65,100], "color": "#FADBD8"},
                ],
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Key risk factors
    st.markdown("### 📌 Key Risk Factors")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Interest Rate", f"{int_rate:.1f}%",
                  delta="High" if int_rate > 20 else "Normal",
                  delta_color="inverse")
    with c2:
        st.metric("Debt-to-Income", f"{dti:.1f}%",
                  delta="Elevated" if dti > 35 else "OK")
    with c3:
        pti = (installment / (annual_inc / 12)) * 100
        st.metric("Payment-to-Income", f"{pti:.1f}%",
                  delta="High" if pti > 25 else "OK")

    if probability > 0.45:
        st.error("⚠️ High default risk detected. Would likely be declined or offered higher rates.")
    elif probability > 0.20:
        st.warning("⚡ Moderate risk. May be approved with standard terms.")
    else:
        st.success("✅ Low default risk. Eligible for competitive interest rates.")

    st.info("💡 Go to **SHAP Explainer** in the sidebar to see which features drove this score.")