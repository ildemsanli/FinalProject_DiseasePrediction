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
        
expander1=st.expander(label='Self-diagnosis could save your life!')

with expander1:
    components.html("""<div class='tableauPlaceholder' 
                    id='viz1650623140832' style='position: 
                        relative'><noscript><a href='#'><img alt='Dashboard 1 ' 
                        src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ca&#47;CausesofMortality_2019&#47;Dashboard1&#47;1_rss.png' 
                        style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
                        <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='CausesofMortality_2019&#47;Dashboard1' /><param name='tabs' 
                        value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ca&#47;CausesofMortality_2019&#47;Dashboard1&#47;1.png' /> 
                        <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' 
                        value='yes' /><param name='language' value='en-GB' /><param name='filter' value='publish=yes' /></object></div>               
                        <script type='text/javascript'>                    var divElement = document.getElementById('viz1650623140832');                    
                        var vizElement = divElement.getElementsByTagName('object')[0];                   
                        if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} 
                        else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} 
                        else { vizElement.style.width='100%';vizElement.style.height='727px';}                     
                        var scriptElement = document.createElement('script');                    
                        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
                        vizElement.parentNode.insertBefore(scriptElement, vizElement);                
                        </script>""", width=900, height=1000)
    




