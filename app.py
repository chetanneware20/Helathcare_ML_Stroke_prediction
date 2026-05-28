import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD FILES ----------------
model = joblib.load("stroke_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
bmi_median = joblib.load("bmi_median.pkl")

# ---------------- TITLE ----------------
st.title("Stroke Prediction App")

# ---------------- USER INPUTS ----------------
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
    value=100.0
)

bmi = st.number_input(
    "BMI",
    value=float(bmi_median)
)

smoking_status = st.selectbox(
    "Smoking Status",
    ["formerly smoked", "never smoked", "smokes", "Unknown"]
)

# ---------------- CREATE INPUT DATA ----------------
input_data = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": Residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status
}

input_df = pd.DataFrame([input_data])

# ---------------- ENCODING ----------------
categorical_cols = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

input_df = pd.get_dummies(input_df, columns=categorical_cols)

# ---------------- MATCH TRAINING COLUMNS ----------------
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Keep exact same order
input_df = input_df[feature_columns]

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("High Risk of Stroke")
    else:
        st.success("Low Risk of Stroke")
