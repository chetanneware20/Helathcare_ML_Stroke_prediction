import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Load model safely
MODEL_PATH = "stroke_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file stroke_model.pkl not found!")
    st.stop()

model = joblib.load(MODEL_PATH)

st.title("🩺 Healthcare Stroke Prediction")
st.write("Predict stroke risk using Machine Learning")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])

work_type = st.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
)

Residence_type = st.selectbox(
    "Residence Type",
    ["Urban", "Rural"]
)

avg_glucose_level = st.number_input(
    "Average Glucose Level",
    min_value=50.0,
    max_value=300.0,
    value=100.0
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

smoking_status = st.selectbox(
    "Smoking Status",
    ["formerly smoked", "never smoked", "smokes", "Unknown"]
)

# Encoding
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
residence_map = {"Urban": 1, "Rural": 0}

work_map = {
    "Private": 0,
    "Self-employed": 1,
    "Govt_job": 2,
    "children": 3,
    "Never_worked": 4
}

smoking_map = {
    "formerly smoked": 0,
    "never smoked": 1,
    "smokes": 2,
    "Unknown": 3
}

# Create dataframe
input_data = pd.DataFrame({
    "gender": [gender_map[gender]],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "ever_married": [married_map[ever_married]],
    "work_type": [work_map[work_type]],
    "Residence_type": [residence_map[Residence_type]],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "smoking_status": [smoking_map[smoking_status]]
})

# Prediction
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")

st.markdown("---")
st.caption("Built with Streamlit & Machine Learning")
