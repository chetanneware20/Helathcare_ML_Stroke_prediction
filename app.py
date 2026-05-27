import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

if os.path.exists("stroke_model.pkl"):
    model = joblib.load("stroke_model.pkl")
else:
    st.error("Model file not found!")
# Load model
model = joblib.load("stroke_model.pkl")

st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🩺",
    layout="centered"
)

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
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
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

input_data = pd.DataFrame({
    "gender": [gender_map[gender]],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "ever_married": [married_map[ever_married]],
    "work_type": [work_map[work_type]],
    "Residence_type": [residence_map[residence_type]],
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
