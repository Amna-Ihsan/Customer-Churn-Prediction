import joblib
import streamlit as st
import pandas as pd

model = joblib.load("rf_classifier_model.pkl")
# scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Customer Churn Prediction")

credit_score = st.number_input("Credit Score", min_value=0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18)
tenure = st.number_input("Tenure", min_value=0, max_value=10)
balance = st.number_input("Balance", min_value=0.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_credit_card = st.selectbox("Has Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
est_salary = st.number_input("Estimated Salary", min_value=0.0)

input_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_credit_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": est_salary,

    # Manual one-hot encoding
    "Geography_Germany": 1 if geography == "Germany" else 0,
    "Geography_Spain": 1 if geography == "Spain" else 0,
    "Gender_Male": 1 if gender == "Male" else 0
}])

input_data = input_data.reindex(columns=model_columns, fill_value=0)

if st.button("Predict"):
    # input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("Prediction: Customer may leave")
    else:
        st.success("Prediction: Customer may stay")

    st.write(f"Churn Probability: {prediction}")