import numpy as np
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from streamlit_option_menu import option_menu
from pathlib import Path
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
from streamlit_lottie import st_lottie
import requests
from model import preprocess_user_input, predict_readmission
names = ["ram", "shyam"]
usernames = ["ra", "sha"]
fp = Path(__file__).parent / "hash.pkl"
with fp.open("rb") as file:
    hp = pickle.load(file)
credentials = {"usernames": {}}
for uname, name, pwd in zip(usernames, names, hp):
    user_dict = {"name": name, "password": pwd}
    credentials["usernames"].update({uname: user_dict})
 
authenticator = stauth.Authenticate(credentials, "hospital_readmission", "abcdef", cookie_expiry_days=0)
 
name, authentication_status, username = authenticator.login('main', fields = {'Form name': 'Login'})
if authentication_status ==False:
    st.error("incorrect details")
if authentication_status:
    st.sidebar.title(f"welcome, {name}!!!")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    s=pd.read_csv("hospital_readmissions.csv")
    df=pd.DataFrame(s)
    with st.sidebar:
        s=option_menu(menu_title="CHOOSE",options=['Home','Data','Predict'],icons=['house','activity','search'],menu_icon="hospital")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    if s=='Home':
        st.title("HOSTPITAL READMISSION")
        def lots(url: str):
            g=requests.get(url)
            if g.status_code !=200:
                return None
            return g.json()
        lotss=lots("https://lottie.host/62f7926f-0d83-488d-bdc6-0a4c43ba00cd/iO81n952pO.json")
        st_lottie(lotss)
 
       
    if s=='Data':
        st.title("Lets understand data")
        x=st.number_input("enter your first number",min_value=0,max_value=len(df)-1)
        y=st.number_input("enter your second number",min_value=x,max_value=len(df)-1)
        new_df=df.iloc[x:y+1]
        plt.bar(new_df['age'],new_df['time_in_hospital'])
        plt.xlabel('Age')
        plt.ylabel('Time in Hospital')
        st.pyplot()
 
 
 
    if s == 'Predict':
        st.title("Let's Predict")
 
        # User input fields
        age = st.selectbox("Enter the person's age:", ['40-50', '50-60', '60-70','70-80', '80-90', '90-100'])
        time_in_hospital = st.number_input("Enter the time spent in hospital:", min_value=0)
        n_lab_procedures = st.number_input("Enter the number of lab procedures performed:", min_value=0)
        n_procedures = st.number_input("Enter the number of procedures performed:", min_value=0)
        n_medications = st.number_input("Enter the number of medications administered:", min_value=0)
        n_outpatient = st.number_input("Enter the number of outpatient visits in the year before hospital stay:", min_value=0)
        n_inpatient = st.number_input("Enter the number of inpatient visits in the year before hospital stay:", min_value=0)
        n_emergency = st.number_input("Enter the number of visits to the emergency room in the year before hospital stay:", min_value=0)
        medical_specialty = st.selectbox("Select medical specialty:", ['Missing', 'Other', 'InternalMedicine', 'Family/GeneralPractice', 'Cardiology', 'Surgery', 'Emergency/Trauma'])
        diag_1 = st.selectbox("Select diagnosis 1:", ['Circulatory', 'Other', 'Injury', 'Digestive', 'Respiratory', 'Diabetes', 'Musculoskeletal', 'Missing'])
        diag_2 = st.selectbox("Select diagnosis 2:", ['Respiratory', 'Other', 'Circulatory', 'Injury', 'Diabetes', 'Digestive', 'Musculoskeletal', 'Missing'])
        diag_3 = st.selectbox("Select diagnosis 3:", ['Other', 'Circulatory', 'Diabetes', 'Respiratory', 'Injury', 'Musculoskeletal', 'Digestive', 'Missing'])
        glucose_test = st.selectbox("Glucose test:", ['no', 'normal', 'high'])
        A1Ctest = st.selectbox("A1C test:", ['no', 'normal', 'high'])
        change = st.radio("Change:", ['Yes', 'No'])
        diabetes_med = st.radio("Diabetes medication:", ['Yes', 'No'])
 
        # Submit button
            # Prepare input data for prediction
        user_input = {
                'age': age,
                'time_in_hospital': time_in_hospital,
                'n_lab_procedures': n_lab_procedures,
                'n_procedures': n_procedures,
                'n_medications': n_medications,
                'n_outpatient': n_outpatient,
                'n_inpatient': n_inpatient,
                'n_emergency': n_emergency,
                'medical_specialty': medical_specialty,
                'diag_1': diag_1,
                'diag_2': diag_2,
                'diag_3': diag_3,
                'glucose_test': glucose_test,
                'A1Ctest': A1Ctest,
                'change': change,
                'diabetes_med': diabetes_med
        }
 
        # Perform prediction using the loaded model
   
        print(user_input)
 
        if st.button("Predict"):
        # Preprocess user input
           
            ans = predict_readmission(user_input)
 
            # Display prediction result
            if ans == 1:
                st.success("Predicted Readmission: Yes")
            else:
                st.success("Predicted Readmission: No")
    authenticator.logout("Logout","sidebar")
   