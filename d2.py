#sore_throat,fever,swelling_of_body,dizziness,headache,bodyache,rash,fatigue,chills,muscleache,coughing
import pandas as pd
import csv
import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
from sklearn import preprocessing
from sklearn import utils
sns.set()

df = pd.read_csv('./oasis_longitudinal.csv')
print(df.head())
Gen={'M': 0,'F': 1} 
df.Gen= [Gen[item] for item in df.Gen] 

df = df.fillna(0)
print(df)

Y = df['CDR'].values # Target for the model
X = df[['Gen','Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(Y)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(Y))
print(utils.multiclass.type_of_target(Y.astype('int')))
print(utils.multiclass.type_of_target(training_scores_encoded))
lr=linear_model.LogisticRegression()
lr.fit(X,training_scores_encoded)
a=[0,80,12,4,20,1494,0.661,1.175]
b=lr.predict([a])
if(b==0):
    print("Normal")
    print("No Need")
elif(b==1):
    print("Mild")

elif(b==2):
    print("Modrate")
else:
    print("sevear")            
