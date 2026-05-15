import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import hashlib

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Insurance Premium Predictor",
    layout="centered"
)

st.title(" Insurance Premium Predictor")
st.write("App running ✅")

import sklearn
st.write("sklearn version:", sklearn.__version__)
with open("model/final_pipeline.pkl", "rb") as f:
    st.write("MODEL HASH:", hashlib.md5(f.read()).hexdigest())

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "model", "final_pipeline.pkl")
    return joblib.load(path)

pipeline = load_model()

st.success("Model loaded successfully ✅")

# ============================================
# USER INPUTS
# ============================================
age = st.number_input("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.number_input("Annual Income", 10000, 1000000, 50000)
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
dependents = st.number_input("Number of Dependents", 0, 10, 1)
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
health = st.slider("Health Score", 0, 100, 70)
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
policy = st.selectbox("Policy Type", ["Basic", "Premium", "Comprehensive"])
claims = st.number_input("Previous Claims", 0, 20, 0)
vehicle_age = st.number_input("Vehicle Age", 0, 30, 5)
credit = st.number_input("Credit Score", 300, 900, 650)
insurance_duration = st.number_input("Insurance Duration", 0, 20, 5)
smoking = st.selectbox("Smoking Status", ["Yes", "No"])
exercise = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])
policy_date = st.date_input("Policy Start Date")

policy_year = policy_date.year
policy_month = policy_date.month

# ============================================
# PREDICTION
# ============================================
if st.button("🚀 Predict Premium"):

    try:

        # CREATE DATAFRAME
        data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Annual Income": [income],
            "Marital Status": [marital],
            "Number of Dependents": [dependents],
            "Education Level": [education],
            "Occupation": [occupation],
            "Health Score": [health],
            "Location": [location],
            "Policy Type": [policy],
            "Previous Claims": [claims],
            "Vehicle Age": [vehicle_age],
            "Credit Score": [credit],
            "Insurance Duration": [insurance_duration],
            "Smoking Status": [smoking],
            "Exercise Frequency": [exercise],
            "Property Type": [property_type],
            "policy_year": [policy_year],
            "policy_month": [policy_month]
        })
        data = data.astype({
        "Age": float,
        "Annual Income": float,
        "Number of Dependents": float,
        "Health Score": float,
        "Previous Claims": float,
        "Vehicle Age": float,
        "Credit Score": float,
        "Insurance Duration": float,
        "policy_year": float,
        "policy_month": float
        })

         

        # ========================================
        # PREDICT
        # ========================================
        prediction = pipeline.predict(data)
        prediction = float(prediction[0])

        st.success("Prediction generated successfully 🎯")

        st.metric(
            label="💰 Estimated Premium",
            value=f"₹ {prediction:,.2f}"
        )

    except Exception as e:
        st.error(f"Error: {e}")