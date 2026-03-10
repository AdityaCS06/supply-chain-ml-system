import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Supply Chain AI Dashboard", layout="wide")

st.title("📦 AI-Powered Supply Chain Optimization")

# Load dataset
data = pd.read_csv("data/raw/supply_chain.csv")

# ---------------- KPI CARDS ----------------
st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)

avg_price = data["Price"].mean()
avg_demand = data["Number of products sold"].mean()
avg_stock = data["Stock levels"].mean()

col1.metric("Average Product Price", f"{avg_price:.2f}")
col2.metric("Average Demand", f"{avg_demand:.0f} units")
col3.metric("Average Stock Level", f"{avg_stock:.0f}")

st.divider()

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🔮 Demand Prediction", "🚚 Supply Chain Risk"])

# ---------------- TAB 1 : ANALYTICS ----------------
with tab1:

    st.subheader("Demand Distribution")

    fig = px.histogram(
        data,
        x="Number of products sold",
        nbins=20,
        title="Demand Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price vs Demand")

    fig2 = px.scatter(
        data,
        x="Price",
        y="Number of products sold",
        title="Price vs Demand Relationship"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Products by Location")

    fig3 = px.bar(
        data,
        x="Location",
        y="Number of products sold",
        title="Demand by Location"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ---------------- TAB 2 : DEMAND PREDICTION ----------------
with tab2:

    st.subheader("Predict Product Demand")

    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Price")

    with col2:
        availability = st.number_input("Availability")

    col3, col4 = st.columns(2)

    with col3:
        stock_levels = st.number_input("Stock Levels")

    with col4:
        lead_times = st.number_input("Lead Time")

    if st.button("Predict Demand"):

        response = requests.post(
            "http://127.0.0.1:8000/predict-demand",
            json={
                "price": price,
                "availability": availability,
                "stock_levels": stock_levels,
                "lead_times": lead_times
            }
        )

        result = response.json()

        st.success(f"Predicted Demand: {result['prediction'][0]:.2f} units")

# ---------------- TAB 3 : RISK ANALYSIS ----------------
with tab3:

    st.subheader("Supply Chain Risk Indicator")

    avg_lead_time = data["Lead times"].mean()

    if avg_lead_time > 20:
        st.error("⚠ High Supply Chain Risk (Long Lead Times)")
    else:
        st.success("✅ Supply Chain Operating Normally")

    st.write("Average Lead Time:", round(avg_lead_time, 2))

    st.subheader("Top 10 High Demand Products")

    top_products = data.sort_values(
        "Number of products sold", ascending=False
    ).head(10)

    st.dataframe(top_products[["SKU", "Number of products sold", "Stock levels"]])

# ---------------- MODEL INFO ----------------
st.divider()
st.subheader("Model Information")

st.write("Model Used: Random Forest Regressor")
st.write("Target Variable: Number of products sold")