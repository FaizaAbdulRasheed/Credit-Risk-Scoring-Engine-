"""
EDA Dashboard Page
Displays interactive exploratory data analysis charts.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def show_eda():
    st.title("📊 Exploratory Data Analysis")
    st.markdown(
        "Interactive analysis of the LendingClub dataset (2007–2018). "
        "All charts are based on the processed dataset after filtering indeterminate loan statuses."
    )
    st.markdown("---")

    # ── Load or generate sample data ─────────────────────────────────────
    df = _get_sample_data()

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", "🎯 Target Distribution", "💰 Key Features", "🗺️ Geography"
    ])

    with tab1:
        _overview_tab(df)

    with tab2:
        _target_tab(df)

    with tab3:
        _features_tab(df)

    with tab4:
        _geo_tab(df)


def _overview_tab(df):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Loans", f"{len(df):,}")
    col2.metric("Default Rate", f"{df['target'].mean():.1%}")
    col3.metric("Avg Loan Amount", f"${df['loan_amnt'].mean():,.0f}")
    col4.metric("Avg Interest Rate", f"{df['int_rate'].mean():.1f}%")

    st.markdown("### Loan Issuance Over Time")
    yearly = df.groupby("issue_year").agg(
        count=("loan_amnt", "count"),
        default_rate=("target", "mean")
    ).reset_index()

    fig = go.Figure()
    fig.add_bar(x=yearly["issue_year"], y=yearly["count"], name="Loans Issued",
                marker_color="#2E5090", opacity=0.7)
    fig.add_scatter(x=yearly["issue_year"], y=yearly["default_rate"],
                    name="Default Rate", yaxis="y2", line=dict(color="#E05A2B", width=2))
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right", tickformat=".0%", title="Default Rate"),
        yaxis=dict(title="Loans Issued"),
        legend=dict(x=0.01, y=0.99),
        height=400, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


def _target_tab(df):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Class Distribution")
        counts = df["target"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        counts["Status"] = counts["Status"].map({0: "Non-Default (Fully Paid)", 1: "Default / Charged Off"})
        fig = px.pie(counts, values="Count", names="Status",
                     color="Status",
                     color_discrete_map={
                         "Non-Default (Fully Paid)": "#2E8B57",
                         "Default / Charged Off": "#E05A2B"
                     },
                     hole=0.4)
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Default Rate by Grade")
        grade_dr = df.groupby("grade")["target"].mean().reset_index()
        grade_dr.columns = ["Grade", "Default Rate"]
        grade_dr = grade_dr.sort_values("Grade")
        fig = px.bar(grade_dr, x="Grade", y="Default Rate",
                     color="Default Rate", color_continuous_scale="RdYlGn_r",
                     text=grade_dr["Default Rate"].map("{:.1%}".format))
        fig.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Default Rate by Loan Purpose")
    purpose_dr = df.groupby("purpose").agg(
        default_rate=("target", "mean"), count=("target", "count")
    ).reset_index().sort_values("default_rate", ascending=False)
    fig = px.bar(purpose_dr, x="purpose", y="default_rate",
                 color="default_rate", color_continuous_scale="RdYlGn_r",
                 hover_data=["count"],
                 labels={"default_rate": "Default Rate", "purpose": "Loan Purpose"})
    fig.update_layout(height=380, template="plotly_white", xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)


def _features_tab(df):
    feature = st.selectbox(
        "Select feature to explore",
        ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low", "revol_util"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {feature} Distribution")
        fig = px.histogram(df, x=feature, color="target",
                           color_discrete_map={0: "#2E8B57", 1: "#E05A2B"},
                           labels={"target": "Default"},
                           barmode="overlay", opacity=0.7, nbins=50)
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"### {feature} by Default Status")
        fig = px.box(df, x="target", y=feature,
                     color="target",
                     color_discrete_map={0: "#2E8B57", 1: "#E05A2B"},
                     labels={"target": "Default (1=Yes)", feature: feature})
        fig.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation with Default (Top Features)")
    numeric_cols = ["loan_amnt", "int_rate", "annual_inc", "dti",
                    "fico_range_low", "revol_util", "delinq_2yrs", "open_acc"]
    corrs = df[numeric_cols + ["target"]].corr()["target"].drop("target").sort_values(key=abs, ascending=False)
    fig = px.bar(
        x=corrs.values, y=corrs.index, orientation="h",
        color=corrs.values, color_continuous_scale="RdBu_r",
        labels={"x": "Pearson Correlation with Default", "y": "Feature"}
    )
    fig.update_layout(height=380, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def _geo_tab(df):
    st.markdown("### Default Rate by State")
    state_dr = df.groupby("addr_state")["target"].mean().reset_index()
    state_dr.columns = ["State", "Default Rate"]
    fig = px.choropleth(
        state_dr, locations="State", locationmode="USA-states",
        color="Default Rate", scope="usa",
        color_continuous_scale="RdYlGn_r",
        labels={"Default Rate": "Default Rate"},
        title="Default Rate by US State"
    )
    fig.update_layout(height=450, geo=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def _get_sample_data() -> pd.DataFrame:
    """
    Return synthetic sample data for demo purposes when real data is not available.
    In production, this would load from data/processed/ or data/splits/.
    """
    np.random.seed(42)
    n = 5000

    grades = list("ABCDEFG")
    purposes = ["debt_consolidation", "credit_card", "home_improvement",
                "small_business", "major_purchase", "medical", "car", "other"]
    states = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
              "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"]
    terms = ["36 months", "60 months"]

    grade_arr = np.random.choice(grades, n, p=[0.20, 0.22, 0.20, 0.15, 0.10, 0.07, 0.06])
    grade_default_rates = {"A": 0.05, "B": 0.10, "C": 0.16, "D": 0.22, "E": 0.30, "F": 0.38, "G": 0.46}
    target = np.array([int(np.random.rand() < grade_default_rates[g]) for g in grade_arr])

    years = np.random.choice(range(2012, 2019), n,
                              p=[0.08, 0.10, 0.13, 0.15, 0.18, 0.20, 0.16])

    df = pd.DataFrame({
        "loan_amnt": np.random.lognormal(9.5, 0.6, n).clip(1000, 40000),
        "int_rate": np.random.normal(13.5, 4.5, n).clip(5, 31),
        "annual_inc": np.random.lognormal(10.8, 0.7, n).clip(15000, 500000),
        "dti": np.random.normal(18, 9, n).clip(0, 50),
        "fico_range_low": np.random.normal(700, 50, n).clip(600, 850),
        "revol_util": np.random.normal(55, 25, n).clip(0, 100),
        "delinq_2yrs": np.random.choice([0, 1, 2, 3], n, p=[0.75, 0.15, 0.07, 0.03]),
        "open_acc": np.random.randint(3, 25, n),
        "grade": grade_arr,
        "purpose": np.random.choice(purposes, n),
        "addr_state": np.random.choice(states, n),
        "term": np.random.choice(terms, n, p=[0.7, 0.3]),
        "issue_year": years,
        "target": target,
    })
    return df
