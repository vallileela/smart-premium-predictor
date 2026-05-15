# ============================================
# 📦 IMPORTS (lightweight only)
# ============================================

import streamlit as st
import os
import hashlib

# ============================================
# ⚙️ PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Insurance Premium Predictor",
    layout="centered"
)

st.title(" Insurance Premium Predictor")
st.write("App reached UI ✅")
st.write("Model is loading... please wait ⏳")

# ============================================
# 📂 LOAD SAVED PIPELINE (CACHED)
# ============================================

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "final_pipeline.pkl"
)

@st.cache_resource
def load_model():
    import joblib
    import os

    path = os.path.join(os.path.dirname(__file__), "model", "final_pipeline.pkl")

    with open(MODEL_PATH, "rb") as f:
        st.write("Model hash:", hashlib.md5(f.read()).hexdigest())

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")

    return joblib.load(path)

pipeline = load_model()




st.success("✅ Model loaded successfully!")


# ============================================
# 📝 USER INPUTS
# ============================================

age = st.number_input("Age", 18, 100, 30)

gender = st.selectbox("Gender", ["Male", "Female"])

income = st.number_input("Annual Income", 10000, 1000000, 50000)

marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

dependents = st.number_input("Number of Dependents", 0, 10, 1)

education = st.selectbox(
    "Education Level",
    ["High School", "Bachelor's", "Master's", "PhD"]
)

occupation = st.selectbox(
    "Occupation",
    ["Employed", "Self-Employed", "Unemployed"]
)

health = st.slider("Health Score", 0, 100, 70)

location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])

policy = st.selectbox(
    "Policy Type",
    ["Basic", "Premium", "Comprehensive"]
)

claims = st.number_input("Previous Claims", 0, 20, 0)

vehicle_age = st.number_input("Vehicle Age", 0, 30, 5)

credit = st.number_input("Credit Score", 300, 900, 650)

insurance_duration = st.number_input("Insurance Duration (Years)", 0, 20, 5)

smoking = st.selectbox("Smoking Status", ["Yes", "No"])

exercise = st.selectbox(
    "Exercise Frequency",
    ["Daily", "Weekly", "Monthly", "Rarely"]
)

property_type = st.selectbox(
    "Property Type",
    ["House", "Apartment", "Condo"]
)

policy_date = st.date_input("Policy Start Date")

# ============================================
# ⚙️ FEATURE ENGINEERING
# ============================================

policy_year = policy_date.year
policy_month = policy_date.month

# ============================================
# 🚀 PREDICTION
# ============================================

if st.button("🚀 Predict Premium"):

    try:
        import pandas as pd

        # Create input dataframe
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

        # Prediction
        prediction = pipeline.predict(data)
        prediction = float(prediction[0])

        # Output
        st.success("✅ Prediction Generated Successfully")

        st.metric(
            label="💰 Estimated Premium",
            value=f"₹ {prediction:,.2f}"
        )

    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")