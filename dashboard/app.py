import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/dashboard_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("dashboard/model.pkl")

@st.cache_resource
def load_columns():
    return joblib.load("dashboard/columns.pkl")

df = load_data()
model = load_model()
model_cols = load_columns()

st.title("Real-Time Fraud Detection Dashboard")

st.sidebar.header("Filters")

min_amt, max_amt = st.sidebar.slider("Transaction Amount Range", 0, 10000, (0, 5000))

risk_filter = st.sidebar.selectbox("Risk Tier", ["All", "Critical Risk", "Suspicious", "Clear"])

df = df[(df["TransactionAmt"] >= min_amt) & (df["TransactionAmt"] <= max_amt)]

if risk_filter != "All":
    df = df[df["RiskTier"] == risk_filter]

page = st.sidebar.radio("Navigation", ["Overview", "Explorer", "SHAP Explainer"])


if page == "Overview":

    total = len(df)
    fraud = df["true_label"].sum()
    fraud_rate = (fraud / total) * 100 if total > 0 else 0
    avg_fraud_amt = df[df["true_label"] == 1]["TransactionAmt"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", total)
    col2.metric("Fraud Cases", fraud)
    col3.metric("Fraud Rate %", round(fraud_rate, 2))
    col4.metric("Avg Fraud Amount", round(avg_fraud_amt, 2))

    fig = px.histogram(df, x="TransactionAmt", color="RiskTier",
                       title="Transaction Distribution by Risk Tier",
                       color_discrete_map={"Critical Risk": "lightcoral", "Suspicious": "palegoldenrod", "Clear": "lightblue"}                 
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(x=df["RiskTier"].value_counts().index,
                  y=df["RiskTier"].value_counts().values,
                  title="Risk Tier Distribution",
                  color=df["RiskTier"].value_counts().index,
                  color_discrete_map={"Critical Risk": "lightcoral", "Suspicious": "palegoldenrod", "Clear": "lightblue"}
    )
    st.plotly_chart(fig2)


elif page == "Explorer":

    st.header("Transaction Explorer")

    txn_id = st.text_input("Enter Transaction Id")

    if txn_id.isdigit():

        txn_id = int(txn_id)

        row = df.iloc[[txn_id]].copy()
        row_model = row.reindex(columns=model_cols, fill_value=0)

        prob = model.predict_proba(row_model)[0][1]

        st.write(row)
        st.metric("Fraud Probability", round(prob, 4))

        if prob >= 0.75:
            st.error("CRITICAL RISK")
        elif prob >= 0.40:
            st.warning("SUSPICIOUS TRANSACTION")
        else:
            st.success("CLEAR TRANSACTION")

    st.subheader("Sample Data")
    st.dataframe(df.head(100))

    fig = px.scatter(df, x="TransactionAmt", y="HourOfDay",
                     color="RiskTier",
                     title="Transaction Patterns",
                        color_discrete_map={"Critical Risk": "lightcoral", "Suspicious": "palegoldenrod", "Clear": "lightblue"},
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "SHAP Explainer":

    st.header("SHAP Explanation")

    idx = st.number_input("Enter Transaction Id", min_value=0, max_value=len(df)-1, step=1)

    row = df.iloc[[idx]].copy()
    row_model = row.reindex(columns=model_cols, fill_value=0)

    prob = model.predict_proba(row_model)[0][1]

    st.write("Selected Transaction:")
    st.dataframe(row)

    st.metric("Fraud Probability", round(prob, 4))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_model)

    st.subheader("SHAP Waterfall Plot")

    fig, ax = plt.subplots()

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=row_model.iloc[0]
        ),
        show=False
    )

    st.pyplot(fig)

    st.subheader("Risk Assessment")
    transaction_amt = row["TransactionAmt"].values[0]
    hour= row["HourOfDay"].values[0]
    top_features = np.argsort(np.abs(shap_values[0]))[-3:][::-1]
    top_feature = row_model.columns[top_features[0]]
    st.write(
        f"""
        The transaction has a predicted fraud probability of {round(prob, 4)}%.

        The most influential feature is '{top_feature}'.

        Transaction amount is {transaction_amt} and it occurred at hour {hour}.

        High values of '{top_feature}' contribute to higher fraud risk, while low values contribute to lower risk.
        """
    )

    if prob > 0.75:
        st.error("High Risk Transaction")
    elif prob > 0.40:
        st.warning("Moderate Risk Transaction")
    else:
        st.success("Low Risk Transaction")
