import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Supply Chain AI System", layout="wide")

# Load dataset
data = pd.read_csv("data/raw/supply_chain.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Demand Prediction", "Supply Chain Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("Target: Demand Forecast")

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.title("📦 Supply Chain Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Price", f"{data['Price'].mean():.2f}")
    col2.metric("Average Demand", f"{data['Number of products sold'].mean():.0f}")
    col3.metric("Average Stock", f"{data['Stock levels'].mean():.0f}")

    st.subheader("Demand Distribution")

    fig = px.histogram(
        data,
        x="Number of products sold",
        title="Demand Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- DEMAND PREDICTION ----------------
elif page == "Demand Prediction":

    st.title("🔮 Demand Prediction")

    # Initialize session state
    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Price", min_value=0.0)

    with col2:
        availability = st.number_input("Availability", min_value=0)

    col3, col4 = st.columns(2)

    with col3:
        stock_levels = st.number_input("Stock Levels", min_value=0)

    with col4:
        lead_times = st.number_input("Lead Time", min_value=0)

    if st.button("Predict Demand"):

        with st.spinner("Predicting demand..."):

            try:
                response = requests.post(
                    # "http://127.0.0.1:8000/predict-demand",
                    "https://supply-chain-ml-system-production.up.railway.app/predict-demand",
                    json={
                        "price": price,
                        "availability": availability,
                        "stock_levels": stock_levels,
                        "lead_times": lead_times
                    }
                )

                if response.status_code == 200:

                    result = response.json()
                    st.session_state.prediction = result["prediction"][0]

                else:
                    st.error("API error. Check FastAPI server.")

            except:
                st.error("API server not running. Start FastAPI first.")

    # Show prediction if available
    if st.session_state.prediction is not None:

        prediction = st.session_state.prediction

        st.success(f"Predicted Demand: {prediction:.2f} units")

        forecast_data = pd.DataFrame({
            "Month": ["Jan", "Feb", "Mar", "Apr"],
            "Demand Forecast": [
                prediction * 0.9,
                prediction,
                prediction * 1.05,
                prediction * 1.1
            ]
        })

        fig = px.line(
            forecast_data,
            x="Month",
            y="Demand Forecast",
            title="Demand Forecast Trend"
        )

        st.plotly_chart(fig)

# ---------------- ANALYTICS ----------------
elif page == "Supply Chain Analytics":

    st.title("📊 Supply Chain Analytics")

    st.subheader("Price vs Demand")

    fig = px.scatter(
        data,
        x="Price",
        y="Number of products sold",
        title="Price vs Demand"
    )

    st.plotly_chart(fig)

    st.subheader("Demand by Location")

    fig2 = px.bar(
        data,
        x="Location",
        y="Number of products sold",
        title="Demand by Location"
    )

    st.plotly_chart(fig2)

    st.subheader("Top 10 High Demand Products")

    top_products = data.sort_values(
        "Number of products sold",
        ascending=False
    ).head(10)

    st.dataframe(
        top_products[["SKU", "Number of products sold", "Stock levels"]]
    )