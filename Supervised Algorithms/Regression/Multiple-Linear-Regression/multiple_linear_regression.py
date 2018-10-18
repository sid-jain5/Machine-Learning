# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:26:47 2018

@author: siddhant
"""
#Multiple Linear Regression

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Imporing the Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] =  labelEncoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoidng the dummy variable trap (here model will take care of the trap, but we can do it manually)
X = X[:, 1:]

#Splitting Train/Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Fitting Multiple linear regression model to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test data result
y_pred = regressor.predict(X_test)

#cannot draw graph because 5 columns mean 5 dimensions

#optimal model using backward elimination
import statsmodels.formula.api as sm
#backward elim eq is y = b0 + b1x1 + b2 x2 + b3x3 ..... + bn xn
# to takw b0 value we take a x0 = 1 column
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regresoor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresoor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regresoor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresoor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regresoor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresoor_OLS.summary()

X_opt = X[:, [0,3,5]]
regresoor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresoor_OLS.summary()

X_opt = X[:, [0,3]]
regresoor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresoor_OLS.summary()