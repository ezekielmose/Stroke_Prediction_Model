# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 21:56:46 2025

@author: Alvine
"""

import numpy as np
import pandas as pd
import streamlit as st
import requests
import pickle

urls = "https://github.com/ezekielmose/Stroke_Prediction_Model/blob/main/strock_model.sav"


loaded_model = requests.get(urls)


#save the loaded model to a tempolary file

with open("strock_model.sav", 'wb') as f:
    
    pickle.dump(loaded_model, f)
    
# loading the saved model from the tempolary file

with open("strock_model.sav", 'rb') as f:
    
    now_loaded_model = pickle.load(f)
    
    
    
def strock_predictor (input_data):
    
   input_data_to_array =  np.array(input_data)
   
   
   input_data_reshaped = input_data_to_array.reshape(1,-1)
   
   predictions = now_loaded_model.predict(input_data_reshaped)
   
   
   
   if predictions [0]==0:
       print("The patient is not at risk")
       
   else:
        print("The patient is most likely to suffer from strock")
        
        
        
        
def mains():
    st.title ("Stroke Prediction Model")
    
    gender = st.text_input("What is the Gender (0 - Female and 1 - male)")
    age = st.text_input("Enter the age")
    hypertension = st.text_input("0 for -ve and 1 for +ve")
    heart_disease = st.text_input("0 for -ve and 1 for +ve")
    ever_married = st.text_input("0 for No and 1 for Yes")
    work_type = st.text_input("0 for private and 1 for self employded, 2 for children, 3 for gov job, and 4 for Never_worked")
    Residence_type = st.text_input ("0 for ubarn  and 1 for rural")
    avg_glucose_level = ("Enter any value as per the measurements")
    bmi =st.text_input ("Enter any value as per the measurements")
    smoking_status =st.text_input("0 for never smoked, 1 for Unknown, 2 for formerly smoked, 3 for smokes")
 					
   
    
   
    diagnosis1 = ""
    
    
    if st.button ("CLICK HERE TO PREDICT"):
        diagnosis1 = strock_predictor ([gender, age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status])
        
    st.success(diagnosis1)
