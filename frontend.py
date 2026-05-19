import pandas as pd
import streamlit as st
import joblib
import numpy as np

model=joblib.load('titanic_model.pkl')
le=joblib.load('label_encoder.pkl')

data=pd.read_csv('titanic.csv')

st.title("Survival Prediction")
st.write("Enter Details to Predict Survival")
st.sidebar.header("About the Project")
st.sidebar.info(
    "This app uses a Logical Regression model trained on the Titanic dataset. "
    "It predicts whether a passenger survived or not."
)
st.sidebar.write("Make your prediction below!")
passenger_id = st.number_input("Passenger ID",min_value=1,value=1)

pclass = st.selectbox("Passenger Class",[1, 2, 3])

sex = st.selectbox("Sex",["male", "female"])

age = st.number_input("Age",min_value=0.0,max_value=100.0,value=25.0)

sibsp = st.number_input("Number of Siblings/Spouses aboard (SibSp)",min_value=0,max_value=10,value=0)

parch = st.number_input("Number of Parents/Children aboard (Parch)",min_value=0,max_value=10,value=0)

fare = st.number_input("Fare",min_value=0.0,max_value=1000.0,value=50.0)

embarked = st.selectbox("Embarked Port",["C", "Q", "S"])

if sex=='male':
    sex=1
else:
    sex=0

embarked_dict = {
    "C": 0,
    "Q": 1,
    "S": 2
}
embarked = embarked_dict[embarked]

if st.button('Predict'):
    input_data=np.array([[passenger_id,pclass,sex,age,sibsp,parch,fare,embarked]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Survival:{prediction}")