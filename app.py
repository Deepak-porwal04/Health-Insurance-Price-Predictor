import streamlit as st
import pickle
import numpy as np

# Import the model
pipe = pickle.load(open('i_pipe.pkl','rb'))
data = pickle.load(open('i_data.pkl','rb'))
st.title('Health Insurance Price Predictor')


# Age
age = st.number_input('Enter Age', step=1)

# Gender
sex = st.selectbox('Gender', data['sex'].unique())

# bmi
bmi = st.number_input('Enter BMI')

# children
children = st.number_input('Enter number of children', step=1)

# smoker
smoker = st.selectbox('Are you a smoker', data['smoker'].unique())

# region
region = st.selectbox('Choose your region', data['region'].unique())

if st.button('Predict Price'):

 
    query = np.array([age, sex, bmi, children, smoker, region])
    query = query.reshape(1,6)

    st.title("The predicted price for your Health Insurance : " + str(int(pipe.predict(query)[0])))