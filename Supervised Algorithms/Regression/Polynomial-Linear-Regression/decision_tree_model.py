# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 00:43:24 2018

@author: siddhant
"""

#Decision Tree Regression

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
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

#Fitting Decision Tree regression model to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#predicting results for 6.5 position using Decision Tree regression model
y_pred = regressor.predict(6.5)

"""#Visualising Decision Tree regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or bluff: Decision Tree Regression')
plt.xlabel('pos')
plt.ylabel('salary')
plt.show()"""

#above plot will not work
# TRAP - Decision tree is a non-linear and "non-continuous" model and
#hence plotting it like above will not work as it will draw straight line
#between each interval and consider each point as an interval

#SOLUTION

#Visualising regression results (for higher resolution and smoother curve)
Xgrid = np.arange(min(X), max(X), 0.01)
Xgrid = Xgrid.reshape((len(Xgrid),1))
plt.scatter(X, y, color = 'red')
plt.plot(Xgrid, regressor.predict(Xgrid), color = 'blue')
plt.title('Truth or bluff: Decision tree Regression')
plt.xlabel('pos')
plt.ylabel('salary')
plt.show()

#plot infeasible for more dimensions