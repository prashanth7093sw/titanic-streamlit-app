
import streamlit as st
import numpy as np
import joblib

# Load the trained model and label encoder
model = joblib.load("logistic_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🚢 Titanic Survival Prediction")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Encode inputs
sex_encoded = label_encoder.transform([sex])[0]
embarked_encoded = label_encoder.transform([embarked])[0]

# Make prediction
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.write(f"🎯 Prediction: **{'Survived' if prediction == 1 else 'Did Not Survive'}**")
    st.write(f"📊 Survival Probability: **{probability:.2f}**")
