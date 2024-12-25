import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl')  # Trained Random Forest model
scaler = joblib.load('scaler.pkl')  # StandardScaler used during training
expected_features = joblib.load('features.pkl')  # Expected feature names

# App title
st.title("Car Price Prediction App")

# Sidebar inputs for car details
st.sidebar.header("Input Car Details:")
present_price = st.sidebar.number_input("Present Price (in lakhs)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, value=20000, step=1000)
age = st.sidebar.number_input("Car Age (in years)", min_value=0, max_value=50, value=5, step=1)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# Encoding categorical inputs
fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
fuel_type_cng = 1 if fuel_type == "CNG" else 0
seller_type_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

# Prepare input data as a dictionary
input_data = {
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Age': [age],
    'Fuel_Type_Diesel': [fuel_type_diesel],
    'Fuel_Type_CNG': [fuel_type_cng],
    'Seller_Type_Individual': [seller_type_individual],
    'Transmission_Manual': [transmission_manual]
}

# Create a DataFrame and reindex to match expected features
input_df = pd.DataFrame(input_data)
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# Scale the input data
scaled_input = scaler.transform(input_df)

# Make prediction
predicted_price = model.predict(scaled_input)[0]

# Display the result
st.subheader(f"Predicted Selling Price: ₹{predicted_price:.2f} lakhs")
