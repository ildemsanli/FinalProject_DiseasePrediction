#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:46:53 2022

@author: ildem
"""

#%%

# Model Testing on heart dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from tpot import TPOTClassifier
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer


data_heart=pd.read_csv('data_heart_processed.csv')

data_heart.outcome.value_counts()

data_heart.columns

data_heart.drop('Unnamed: 0', axis=1, inplace=True)


# Split x and y
x=data_heart[[ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
       'thalach', 'exang', 'oldpeak']]


y=data_heart['outcome']


# Model testing with reduced number of features
x_selected=data_heart[[ 'age', 'sex', 'cp', 'trestbps', 'chol']]


# Simplify the 'cp' column pain (enter 1 for values 1, 2, 3) or no pain (0 for value 4)
x_selected['cp']=np.where(x_selected['cp']!=4, 1, x_selected['cp'])

x_selected['cp']=np.where(x_selected['cp']==4, 0, x_selected['cp'])


# TPOT with x_selected - only 5 features included


model=TPOTClassifier(generations=30, population_size=100, 
                       scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)

model.fit(x_selected, y)

model.export('Heart_cp.py')

"""
Best pipeline: ExtraTreesClassifier(Normalizer(MaxAbsScaler(BernoulliNB(input_matrix, alpha=0.001, fit_prior=True)), norm=l1), bootstrap=False, criterion=gini, max_features=0.7500000000000001, min_samples_leaf=8, min_samples_split=4, n_estimators=100)
Out[463]: 
TPOTClassifier(generations=30, n_jobs=-1, random_state=1, scoring='accuracy',
               verbosity=2)
"""

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=BernoulliNB(alpha=0.001, fit_prior=True)),
    MaxAbsScaler(),
    Normalizer(norm="l1"),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.7500000000000001, min_samples_leaf=8, min_samples_split=4, n_estimators=100)
)


# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.2, 
                                                    random_state=42)

m=exported_pipeline.fit(x_train, y_train)

y_pred=m.predict(x_test)
acc_train=exported_pipeline.score(x_train, y_train)
acc_test=exported_pipeline.score(x_test, y_test)

####confusion matrix
matrix=metrics.confusion_matrix(y_test, y_pred)

####precision score
p=metrics.precision_score(y_test, y_pred, average='weighted')

####recall score
r=metrics.recall_score(y_test, y_pred, average='weighted')

####f1 score
f1=metrics.f1_score(y_test, y_pred, average='weighted')


print('\nAccuracy on train:', round(acc_train, 2))
print('\nAccuracy on test:', round(acc_test, 2))
#print('\nConfusion matrix\n', matrix)
print('\nPrecision score:', round(p, 2))
print('\nRecall score:', round(r, 2))
print('\nF1 score:', round(f1, 2))

# Save confusion matrix as an image
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True)
plt.title('Confusion matrix')
plt.savefig('matrix')


"""

Accuracy on train: 0.81

Accuracy on test: 0.77

Confusion matrix
 [[44 16]
 [14 59]]

Precision score: 0.77

Recall score: 0.77

F1 score: 0.77

"""

# Try the model with user input
user_input=[[54,1,1,1300,1600]]

pred=m.predict(user_input)

if pred==1:
    print('Low risk')
elif pred==0:
    print('High risk')
    
# Save the model to use it in streamlit
import pickle as pkl
pkl.dump(m,open("final_model.p","wb"))








