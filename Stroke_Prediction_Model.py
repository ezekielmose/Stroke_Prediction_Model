import numpy as np
import pandas as pd
import streamlit as st
import requests
import pickle
url = "https://raw.githubusercontent.com/ezekielmose/Stroke_Prediction_Model/main/strock_model.sav"
# Download the file
loaded_model = requests.get(url)

# Save the downloaded content to a temporary file
with open('trained_model1.sav', 'wb') as f:
    pickle.dump(loaded_model, f)


# Load the saved model
with open('trained_model1.sav', 'rb') as f:
    loaded_model = pickle.load(f)
    
def strock_predictor (input_data):
   input_data_to_array =  np.array(input_data)
   input_data_reshaped = input_data_to_array.reshape(1, -1)

   
   prediction = loaded_model.predict(input_data_reshaped)
   
   
   if prediction [0]==0:
       print("The patient is not at risk") 
   else:
        print("The patient is most likely to suffer from strock")
        
        
def main():
    st.title ("Stroke Prediction Model")
    
    gender = st.text_input("What is the Gender (0 - Female and 1 - male)")
    age = st.text_input("Enter the age")
    hypertension = st.text_input(" Hypertension 0 for -ve and 1 for +ve")
    heart_disease = st.text_input(" Heart_disease 0 for has and 1 for not ")
    ever_married = st.text_input(" Ever_married 0 for No and 1 for Yes")
    work_type = st.text_input(" Work_type 0 for private and 1 for self employded, 2 for children, 3 for gov job, and 4 for Never_worked")
    Residence_type = st.text_input ("Residence_type 0 for ubarn  and 1 for rural")
    avg_glucose_level = ("Enter any value of (avg_glucose_level) as per the measurements")
    bmi =st.text_input ("Enter any value of (BMI) as per the measurements")
    smoking_status =st.text_input("Smoking_status 0 for never smoked, 1 for Unknown, 2 for formerly smoked, 3 for smokes")
    
    
    gender = pd.to_numeric(gender, errors='coerce')
    age = pd.to_numeric(age, errors='coerce')
    hypertension = pd.to_numeric(hypertension, errors='coerce')
    heart_disease = pd.to_numeric(heart_disease, errors='coerce')
    ever_married = pd.to_numeric(ever_married, errors='coerce')
    Residence_type = pd.to_numeric(Residence_type, errors='coerce')
    avg_glucose_level = pd.to_numeric(avg_glucose_level, errors='coerce')
    bmi = pd.to_numeric(bmi, errors='coerce')
    smoking_status = pd.to_numeric(smoking_status, errors='coerce')
    
    diagnosis1 = ""
    
    if st.button ("CLICK HERE TO PREDICT"):
        diagnosis1 = strock_predictor ([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi,smoking_status])   
    st.success(diagnosis1) 
    
    
# this is to allow our web app to run from anaconda command prompt where the cmd takes the main() only and runs the code   
if __name__ == '__main__':
    main()
