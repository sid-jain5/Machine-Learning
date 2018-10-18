# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:10:08 2018

@author: siddhant
"""
#Polynomial Regression

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Imporing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
#ignore 1st column as it is equivalent to second
# X taken as 1:2 and not just 1 so that
#it is treated as matrix and not a simple array like y
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Splitting Train/Test Set not required because small dataset
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)"""

 
"""#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X, y)

#Fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 2)
#X_poly has three columns 1st coln is for b0 same like in multiple regression
#with all values 1 of x0 , second columns is same as X and third column
#contains square of terms because degree is 2
X_poly = polyReg.fit_transform(X)
linReg2 = LinearRegression()
linReg2.fit(X_poly, y)

#Visualising Linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linReg.predict(X), color = 'blue')
plt.title('Truth or bluff: Linear Regression')
plt.xlabel('pos')
plt.ylabel('salary')
plt.show()

#Visualising Polynomial regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linReg2.predict(polyReg.fit_transform(X)), color = 'blue')
plt.title('Truth or bluff: Polynomial Regression')
plt.xlabel('pos')
plt.ylabel('salary')
plt.show()

# trying same with degree 3

polyReg1 = PolynomialFeatures(degree = 3)
X_poly1 = polyReg1.fit_transform(X)
linReg3 = LinearRegression()
linReg3.fit(X_poly1, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, linReg3.predict(polyReg1.fit_transform(X)), color = 'blue')
plt.title('Truth or bluff: Polynomial Regression')
plt.xlabel('pos')
plt.ylabel('salary')
plt.show() #better accuracy

# trying same with degree 4

polyReg2 = PolynomialFeatures(degree = 4)
X_poly2 = polyReg2.fit_transform(X)
linReg4 = LinearRegression()
linReg4.fit(X_poly2, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, linReg4.predict(polyReg2.fit_transform(X)), color = 'blue')
plt.title('Truth or bluff: Polynomial Regression')
plt.xlabel('pos')
plt.ylabel('salary')
plt.show() #totally accurate

#still problem : straight line between 2 points(eg 1 and 2 or 2 and 3 and so on)
#because increment is of 1, should be reduced to 0.1

Xgrid = np.arange(min(X), max(X), 0.1)
Xgrid = Xgrid.reshape((len(Xgrid), 1))

plt.scatter(X, y, color = 'red')
plt.plot(Xgrid, linReg4.predict(polyReg2.fit_transform(Xgrid)), color = 'blue')
plt.title('Truth or bluff: Polynomial Regression')
plt.xlabel('pos')
plt.ylabel('salary')
plt.show()

#predicting results for 6.5 position using linear regression
linReg.predict(6.5)

#predicting results for 6.5 position using polynomial regression
linReg4.predict(polyReg2.fit_transform(6.5))