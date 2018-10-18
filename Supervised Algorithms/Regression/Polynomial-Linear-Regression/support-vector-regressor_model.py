# -*- coding: utf-8 -*-
"""
Created on Thu May 24 19:40:11 2018

@author: siddhant
"""
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Imporing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Splitting Train/Test Set not required because small dataset
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)"""

 
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X =sc_X.fit_transform(X)
y =sc_y.fit_transform(y)

#Fitting SVR model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#predicting results
#remove feature scaling to get right result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualising SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or bluff: SVR Regression')
plt.xlabel('pos')
plt.ylabel('salary')
plt.show()
