#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:30:55 2022

@author: ildem
"""

#%%
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle as pkl
from PIL import Image
import streamlit.components.v1 as components

m=pkl.load(open("final_model.p","rb"))

#st.map()


def preprocess(age,sex,pain,pressure,chol):   
 
    
    # Pre-processing user input   
    if sex=="Male":
        sex=1 
    else: 
        sex=0
    
    
    if pain=="Yes":
        pain=1
    else: 
        pain=0
   
    
    user_input=[[age,sex,pain,pressure,chol]]

    prediction = m.predict(user_input)

    return prediction


image = Image.open('heart.jpeg')

st.set_page_config(layout="wide", initial_sidebar_state="expanded")



with st.sidebar:
    st.image(image)



st.title('i-Diagnose')

container1 = st.container()
col1, col2 = st.columns(2)

with container1:
    with col1:
        sex=st.radio("Select Gender: ", ('Male', 'Female'))
    with col2:
        pain=st.radio("Do you have chest pain?", ('Yes', 'No'))

container2 = st.container()
col3, col4 = st.columns(2)       

with container1:
    with col3:
        age=st.selectbox('Please enter your age',range(1,120), index=29)
    with col4:
        pressure=st.selectbox('Please enter your blood pressure (mmHg)',range(60,500), index=100)

container3 = st.container()
col5, col6 = st.columns(2)  

with container3:
    with col5:
        chol=st.selectbox('Please enter your cholesterol level (mg/dl)',range(80,1000), index=100)
    
        
prediction=preprocess(age,sex,pain,pressure,chol) 


if st.button("Predict"):    
    if prediction==0:
        st.error('High risk')
    
    else:
        st.success('Low risk')
        
expander1=st.expander(label='More info')

with expander1:
    st.write('Cool')
    




