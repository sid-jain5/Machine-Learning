# -*- coding: utf-8 -*-
"""
Created on Sun May  6 20:04:45 2018

@author: siddhant
"""

#Simple linear Regression

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Imporing the Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting Train/Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

#Fitting SLR model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Visualizing the train data results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Sal vs Exp for Train set')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()

#Visualizing the test data results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Sal vs Exp for Test set')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()

"""#Visualizing the results
plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_test, y_pred, color = 'blue')
plt.title('pred vs actual test set')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()"""