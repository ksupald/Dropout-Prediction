import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('dropout_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎓 Dropout Prediction App")

# Input fields
gpa = st.number_input("GPA (0 - 4.0)", min_value=0.0, max_value=4.0, step=0.01)
attendance = st.slider("Attendance (%)", 0, 100, 75)
age = st.number_input("Age", min_value=15, max_value=35, step=1)
financial_aid = st.selectbox("Receiving Financial Aid?", ["Yes", "No"])
family_support = st.selectbox("Family Support Available?", ["Yes", "No"])

# Convert to numeric
financial_aid_val = 1 if financial_aid == "Yes" else 0
family_support_val = 1 if family_support == "Yes" else 0

# Prepare input
input_data = np.array([[gpa, attendance, age, financial_aid_val, family_support_val]])
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Dropout Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ High risk of dropout! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low risk of dropout. (Probability: {probability:.2f})")
