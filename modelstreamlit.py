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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator

data_heart=pd.read_csv('data_heart_processed.csv')
x=data_heart[[ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
       'thalach', 'exang', 'oldpeak']]


y=data_heart['outcome']

# Model testing with reduced number of features
x_selected=data_heart[[ 'age', 'sex', 'cp', 'trestbps', 'chol']]


# Simplify the 'cp' column pain (enter 1 for values 1, 2, 3) or no pain (0 for value 4)
x_selected['cp']=np.where(x_selected['cp']!=4, 1, x_selected['cp'])

x_selected['cp']=np.where(x_selected['cp']==4, 0, x_selected['cp'])

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=BernoulliNB(alpha=0.001, fit_prior=True)),
    MaxAbsScaler(),
    Normalizer(norm="l1"),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.7500000000000001, min_samples_leaf=8, min_samples_split=4, n_estimators=100)
)

x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.2, 
                                                    random_state=42)

m=exported_pipeline.fit(x_train, y_train)


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



st.set_page_config(layout="wide", initial_sidebar_state="expanded")



with st.sidebar:
    st.image('heartt.jpeg')



st.title('i-Diagnose')

st.write('Find out your risk of heart diseases')

container1 = st.container()
col1, col2 = st.columns(2)

with container1:
    with col1:
        sex=st.radio("Select sex: ", ('Male', 'Female'))
    with col2:
        pain=st.radio("Do you have chest pain?", ('Yes', 'No'))

container2 = st.container()
col3, col4 = st.columns(2)       

with container1:
    with col3:
        age=st.selectbox('Please enter your age',range(1,120), index=59)
        

container3 = st.container()
col5, col6 = st.columns(2)  

with container3:
    with col5:
        pressure=st.slider('Please enter your blood pressure (mmHg)', 60, 200)
    with col6:
        chol=st.slider('Please enter your cholesterol level (mg/dl)',70, 500)
    
        
prediction=preprocess(age,sex,pain,pressure,chol) 


if st.button("Predict"):    
    if prediction==0:
        st.error('High risk')
    
    else:
        st.success('Low risk')
        

