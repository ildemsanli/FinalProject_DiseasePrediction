#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:32:19 2022

@author: ildem
"""

#%%
import numpy as np
import pandas as pd 
import os

#----------------------------------------------

# Import heart disease datasets

 #3 (age)
 #4 (sex)
 #9 (cp) chest pain Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic
 #10 (trestbps)  resting blood pressure (in mm Hg on admission to the hospital)
 #12 (chol)   serum cholestoral in mg/dl
 #16 (fbs)    (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
 #19 (restecg)  resting electrocardiographic results
                 # Value 0: normal
                 #Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
                 # Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
 #32 (thalach)  thalach: maximum heart rate achieved
 #38 (exang)    exercise induced angina (1 = yes; 0 = no)
 #40 (oldpeak)  ST depression induced by exercise relative to rest
 #41 (slope)   the slope of the peak exercise ST segment Value 1: upsloping, Value 2: flat, Value 3: downsloping
 #44 (ca)     number of major vessels (0-3) colored by flourosopy
 #51 (thal)   thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
 #58 (num) (the predicted attribute)  num: diagnosis of heart disease (angiographic disease status), Value 0: < 50% diameter narrowing, Value 1: > 50% diameter narrowing

col_names= ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca', 'thal', 'num']

data_heart=pd.read_csv('longbeach.data', names=col_names, header=None)

data_heart=data_heart.append(pd.read_csv('cleveland.data', names=col_names, header=None))

data_heart=data_heart.append(pd.read_csv('switzerland.data', names=col_names, header=None))

data_heart=data_heart.append(pd.read_csv('hungarian.data', names=col_names, header=None))

data_heart.isna().sum() # No Nan values

for i in data_heart.columns:
    print('\n--', i, '--\n')
    print(data_heart.value_counts(i))
    
# All columns except age, sex, cp and num have missing values entered as '?'
# slope, ca, thal have more than half the values '?', so we drop these columns.
# 79 rows with level 0 in chol column.

data_heart.drop(['slope', 'ca', 'thal'], axis=1, inplace=True)

data_heart.replace('?', np.NaN, inplace=True)

data_heart.chol.replace(0, np.NaN, inplace=True)

data_heart.chol.isna().sum()

data_heart.dropna(axis=0, how='any', inplace=True)

# 7 object columns with numerical data
# Change them to numerical with pd.to_numeric
data_heart.columns

obj_columns=['trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak']

data_heart[obj_columns]=data_heart[obj_columns].apply(pd.to_numeric)

data_heart.dtypes # all numerical 

# Create an outcome column with 0 and 1. All values other than 0 in 'num' replaced by 1.
# Drop the num column.

data_heart['outcome']=data_heart['num']

data_heart['outcome']=np.where(data_heart['outcome']!=0, 1, data_heart['outcome'])

data_heart.drop(['num'], axis=1, inplace=True)

data_heart.to_csv('data_heart_processed.csv')

#-------------------------------------------




