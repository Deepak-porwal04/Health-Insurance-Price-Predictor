# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 22:37:53 2021

@author: Admin
"""

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score


sns.countplot(data['sex'])
sns.countplot(data['smoker'])
sns.countplot(data['region'])

sns.boxplot(data['age'])


# Importing the dataset
data = pd.read_csv('Health_insurance.csv')
data.info()

X = data.drop('charges', axis = 1)
y = data['charges']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False, drop='first'),[1,4,5])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

r2_score(y_test,y_pred)

import pickle

pickle.dump(data,open('i_data.pkl','wb'))
pickle.dump(pipe,open('i_pipe.pkl','wb'))