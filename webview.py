# copy the pickle file and training dataset into a new folder

#Web application development using streamlit
#load the necessary libraries

import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Vehicle Insrance Response Prediction")

# read the dataset to fill the values in input options of each elements
df = pd.read_csv('train_vehicle_insurance.csv')

#create the input elements
Gender = st.selectbox("Gender",pd.unique(df['Gender']))
Vehicle_Age = st.selectbox("Vehicle_Age",pd.unique(df['Vehicle_Age']))
Vehicle_Damage = st.selectbox("Vehicle_Damage",pd.unique(df['Vehicle_Damage']))

#non-categorical columns
Age = st.number_input("Age")
Driving_License = st.number_input("Driving_License")
Region_Code = st.number_input("Region_Code")
Previously_Insured = st.number_input("Previously_Insured")
Annual_Premium = st.number_input("Annual_Premium")
Policy_Sales_Channel = st.number_input("Policy_Sales_Channel")
Vintage = st.number_input("Vintage")


#Map the user input to respective column format
input={
'Gender' : Gender,
'Age' : Age,
'Driving_License' : Driving_License,
'Region_Code' : Region_Code,
'Previously_Insured' : Previously_Insured,
'Vehicle_Age' : Vehicle_Age,
'Vehicle_Damage' : Vehicle_Damage,
'Annual_Premium' : Annual_Premium,
'Policy_Sales_Channel' : Policy_Sales_Channel,
'Vintage' : Vintage
}

#load the model from the pickle file
model = joblib.load('vehicle_insurance_pipeline_model.pkl')

#Action for submit button
if st.button('Predict'):
    X_input = pd.DataFrame(input,index=[0])
    prediction = model.predict(X_input)
    st.write("The predicted value is:")
    st.write(prediction)


#File Uplaod experiment
st.subheader("please upload a csv file fo prediction")
upload_file = st.file_uploader("Choose a file", type=['csv'])

if upload_file is not None:
    df = pd.read_csv(upload_file)

    st.write("File uploaded successfully")
    st.write(df.head(2))

    if st.button("Predict for the uploaded file"):
        df['Response'] = model.predict(df)
        st.write("pediction completed")
        st.write(df.head(2))
        st.download_button(label="Download prediction", 
                           data=df.to_csv(index=False), 
                           file_name="predictions.csv", mime="text/csv")