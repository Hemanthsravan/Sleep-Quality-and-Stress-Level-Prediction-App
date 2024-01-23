import streamlit as st 
import numpy as np 
import pandas as pd
import pickle
import time


def load_model():
  with open("model.pk1", "rb") as file:
    model = pickle.load(file)
  return model

model = load_model()

regressor = model["model"]
gender_le = model['gender_le']
occupation_le = model['occupation_le']
bmiCategory_le = model['bmiCategory_le']
sleepDisorder_le = model['sleepDisorder_le']
scaler = model['scaler']

def show_predict_page():
  st.title("Sleep Quality & Stress Analysis Predictor")

  st.write(""" ### We need some information to predict the stress levels
           """)
  
  gender = ("Male", "Female",)

  occupation = (
    "Nurse",
    "Doctor",
    "Engineer",
    "Lawyer", 
    "Teacher", 
    "Accountant", 
    "Salesperson", 
    "Software Engineer",
    "Scientist", "Sales Representative", "Manager", 
  )

  bmicategory = (
    "Normal", "Overweight", "Obese",
  )

  sleepdisorder = (
    "No Disorder", "Sleep Apnea","Insomnia",
  )

  gender = st.selectbox("Gender", gender)
  age = st.number_input("Enter your age", 0, 70)
  occupation = st.selectbox("Occupation", occupation)
  sleepduration = st.number_input("Sleep Duration", 0, 20)
  qualityofsleep = st.slider("Quality of Sleep", 0, 16, 4)
  physicalactivitylevel = st.slider("Physical Activity Level", 0, 100, 20)
  bmicategory = st.selectbox("BMI Category", bmicategory)
  heartrate = st.slider("Heart Rate", 50,100)
  dailysteps = st.slider("Daily Steps", 0,10000, 1000)
  sleepdisorder = st.selectbox("Sleep Disorder", sleepdisorder)
  bphigh = st.number_input("BP High", 50, 150)
  bplow = st.number_input("BP Low", 50, 100)


  ok = st.button("Calculate Stress Level")

  if ok:
    X = {'Gender': gender,
            'Age': age,
            'Occupation': occupation,
            'Sleep Duration': sleepduration,
            'Quality of Sleep': qualityofsleep,
            'Physical Activity Level': physicalactivitylevel,
            'BMI Category': bmicategory,
            'Heart Rate': heartrate,
            'Daily Steps': dailysteps,
            'Sleep Disorder': sleepdisorder,
            'BP High': bphigh,
            'BP Low': bplow
            }

    X = pd.DataFrame(X, index=[0])

    X['Gender'] = gender_le.fit_transform(X['Gender'])
    X['Occupation'] = occupation_le.fit_transform(X['Occupation'])
    X['BMI Category'] = bmiCategory_le.fit_transform(X['BMI Category'])
    X['Sleep Disorder'] = sleepDisorder_le.fit_transform(X['Sleep Disorder'])

    numerical_features = ['Age',
                            'Sleep Duration',
                            'Quality of Sleep',
                            'Physical Activity Level',
                            'Heart Rate',
                            'Daily Steps',
                            'BP High',
                            'BP Low']

    X[numerical_features] = scaler.transform(X[numerical_features])


    with st.spinner('Wait for prediction...'):
        y_pred = regressor.predict(X)
        time.sleep(2)

    st.subheader(f"The predicted stress level based on your selections is as follows {np.round(y_pred[0],2)}")
    st.write(" ###### (3-4) - Lowest Stress Level ")
    st.write(" ###### (7-8) - Highest Stress Level for a Person ")

    
  

    

