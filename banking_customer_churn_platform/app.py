import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Banking Customer Churn Platform",
    page_icon="🏦",
    layout="wide"
)

DATA_PATH = "data/churn.csv"
MODEL_PATH = "churn_model.pkl"

st.title("🏦 Banking Customer Churn Prediction & Analytics Platform")
st.caption("Machine-learning dashboard for understanding and predicting customer churn.")

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Add your CSV as data/churn.csv.")
    st.stop()

df = pd.read_csv(DATA_PATH)

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict Churn", "High-Risk Customers", "Data Preview"]
)

if menu == "Dashboard":
    st.header("📊 Churn Analytics Dashboard")

    total = len(df)
    churned = int(df["Exited"].sum())
    stayed = total - churned
    churn_rate = churned / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Stayed", f"{stayed:,}")
    c3.metric("Churned", f"{churned:,}")
    c4.metric("Churn Rate", f"{churn_rate:.2f}%")

    left, right = st.columns(2)

    with left:
        status_df = df.copy()
        status_df["Status"] = status_df["Exited"].map({0: "Stayed", 1: "Churned"})
        fig = px.pie(status_df, names="Status", hole=0.45, title="Overall Churn Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        geo = df.groupby("Geography", as_index=False)["Exited"].mean()
        geo["Churn Rate (%)"] = geo["Exited"] * 100
        fig = px.bar(geo, x="Geography", y="Churn Rate (%)", title="Churn Rate by Geography")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            df, x="Age", color="Exited",
            barmode="overlay",
            title="Age Distribution and Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.box(df, x="Exited", y="Balance", title="Account Balance vs Churn")
        st.plotly_chart(fig, use_container_width=True)

    active = df.groupby("IsActiveMember", as_index=False)["Exited"].mean()
    active["Membership"] = active["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    active["Churn Rate (%)"] = active["Exited"] * 100
    fig = px.bar(active, x="Membership", y="Churn Rate (%)", title="Active Membership vs Churn")
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Predict Churn":
    st.header("🔮 Predict Individual Customer Churn")

    if not os.path.exists(MODEL_PATH):
        st.warning("Model not trained yet. Run: python train_model.py")
        st.stop()

    model = joblib.load(MODEL_PATH)

    c1, c2 = st.columns(2)

    with c1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", sorted(df["Geography"].dropna().unique()))
        gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
        age = st.slider("Age", 18, 100, 35)
        tenure = st.slider("Tenure (years)", 0, 10, 5)

    with c2:
        balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
        products = st.slider("Number of Products", 1, 4, 1)
        card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        active = st.selectbox("Active Member?", ["Yes", "No"])
        salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=1000.0)

    input_df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": 1 if card == "Yes" else 0,
        "IsActiveMember": 1 if active == "Yes" else 0,
        "EstimatedSalary": salary
    }])

    if st.button("Predict Churn Risk", type="primary"):
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        st.metric("Churn Probability", f"{probability * 100:.2f}%")
        st.progress(min(max(probability, 0.0), 1.0))

        if probability >= 0.70:
            st.error("🔴 HIGH RISK — This customer has a high probability of leaving.")
        elif probability >= 0.40:
            st.warning("🟠 MEDIUM RISK — This customer should be monitored.")
        else:
            st.success("🟢 LOW RISK — This customer is currently less likely to churn.")

        if prediction == 1:
            st.write("**Model prediction:** Churn")
        else:
            st.write("**Model prediction:** Stay")

elif menu == "High-Risk Customers":
    st.header("🚨 High-Risk Customer Analysis")

    if not os.path.exists(MODEL_PATH):
        st.warning("Train the model first using: python train_model.py")
        st.stop()

    model = joblib.load(MODEL_PATH)

    feature_df = df.drop(
        columns=[c for c in ["RowNumber", "CustomerId", "Surname", "Exited"] if c in df.columns]
    )

    scored = df.copy()
    scored["ChurnProbability"] = model.predict_proba(feature_df)[:, 1]
    scored["RiskLevel"] = pd.cut(
        scored["ChurnProbability"],
        bins=[-0.01, 0.40, 0.70, 1.0],
        labels=["Low", "Medium", "High"]
    )

    high = scored.sort_values("ChurnProbability", ascending=False)
    high["ChurnProbability"] = (high["ChurnProbability"] * 100).round(2)

    st.metric("High-Risk Customers", int((high["RiskLevel"] == "High").sum()))
    st.dataframe(high.head(100), use_container_width=True)

    st.download_button(
        "Download Risk Analysis CSV",
        high.to_csv(index=False).encode("utf-8"),
        "customer_churn_risk_analysis.csv",
        "text/csv"
    )

elif menu == "Data Preview":
    st.header("🗂️ Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Dataset Information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", int(df.isna().sum().sum()))

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
