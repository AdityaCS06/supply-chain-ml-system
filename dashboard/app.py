import streamlit as st
import requests

st.title("Supply Chain Optimization Dashboard")

distance = st.number_input("Distance")
traffic = st.number_input("Traffic Level")
shipping_cost = st.number_input("Shipping Cost")

if st.button("Predict Demand"):
    response = requests.post(
        "http://127.0.0.1:8000/predict-demand",
        json={
            "price": distance,
            "availability": traffic,
            "stock_levels": shipping_cost,
            "lead_times": 7
        }
    )

    st.write(response.json())