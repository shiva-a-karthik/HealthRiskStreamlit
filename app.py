import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Health Risk Predictor", layout="centered")

st.title("Health Risk Predictor")
st.write("Enter your biometric metrics below to assess predicted cardiovascular risk.")

#Load Model
@st.cache_resource
def load_model():
    return joblib.load("health_model.pkl")

model = load_model()

#Input & Validation
age = st.number_input("Enter Age", min_value=1, max_value=120, value=25, step=1)
bmi = st.number_input("Enter BMI", min_value=10.0, max_value=60.0, value=22.5, step=0.1)
blood_pressure = st.number_input("Enter Blood Pressure", min_value=50, max_value=250, value=120, step=1)

#Trigger Prediction
if st.button("Check Risk"):
    features = np.array([[age, bmi, blood_pressure]])
    raw_pred = model.predict(features)[0]
    risk_score = round(max(0.0, min(100.0, raw_pred)), 2)

    #Feedback
    st.subheader("Assessment Result")
    st.metric(label="Calculated Health Risk Index", value=f"{risk_score} / 100")

    if risk_score < 40:
        st.success("Low Risk: Vital indicators are in a healthy range.")
    elif risk_score < 70:
        st.warning("Moderate Risk: Lifestyle modifications or checkup advised.")
    else:
        st.error("High Risk: Biometrics indicate significant clinical strain.")