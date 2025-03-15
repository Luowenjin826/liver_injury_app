import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('random_forest_model.pkl')

# App title and inputs
st.title("Liver Damage After Taking Tigacycline Probability Prediction")
st.subheader("Input Patient Information")

pct = st.number_input("PCT (ng/mL)", value=0.0, format="%.2f")
maintain_dosage = st.number_input("Maintain Dosage (mg/kg)", value=0.0, format="%.2f")
blood_infection = st.selectbox("Blood Infection", ["No", "Yes"])
ventilation = st.selectbox("Mechanical Ventilation", ["No", "Yes"])
chemo = st.selectbox("Chemotherapeutics", ["No", "Yes"])

if st.button("Predict"):
    # Convert categorical variables to binary
    blood_infection_val = 1 if blood_infection == "Yes" else 0
    ventilation_val = 1 if ventilation == "Yes" else 0
    chemo_val = 1 if chemo == "Yes" else 0
    
    # Create input array with proper shape
    input_data = np.array([[pct, maintain_dosage, blood_infection_val, ventilation_val, chemo_val]])
    
    # Predict probability
    try:
        probability = model.predict_proba(input_data)[:, 1][0]
        st.success(f"The probability of liver damage is {probability:.2%}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
