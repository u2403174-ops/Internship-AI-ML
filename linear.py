import pandas as pd 
import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
diabetics=load_diabetes()
data=load_diabetes(as_frame=True).frame
model=joblib.load('diabetes_model.pkl')
scalar=joblib.load('diabetes_scalar.pkl')
st.title("Diabetes Prediction")
st.write("Enter Details to Predict Diabetes")
st.sidebar.header("About the project")
st.sidebar.info(
    "This app uses a Logistic Regression model trained on the Pima Indians Diabetes dataset. "
    "It predicts whether a person has diabetes or not based on various health parameters."
)
st.sidebar.write("Make your prediction below")
age = st.number_input("Age", value=0.0)

sex = st.number_input("Sex", value=0.0)

bmi = st.number_input("BMI", value=0.0)

bp = st.number_input("Blood Pressure", value=0.0)

s1 = st.number_input("S1", value=0.0)

s2 = st.number_input("S2", value=0.0)

s3 = st.number_input("S3", value=0.0)

s4 = st.number_input("S4", value=0.0)

s5 = st.number_input("S5", value=0.0)

s6 = st.number_input("S6", value=0.0)
if st.button('Predict'):
    input_data=np.array([[age,sex,bmi,bp,s1,s2,s3,s4,s5,s6]])
    input_data_scaled=scalar.transform(input_data)
    prediction=model.predict(input_data_scaled)
    st.write(f"Prediction: {prediction}")

