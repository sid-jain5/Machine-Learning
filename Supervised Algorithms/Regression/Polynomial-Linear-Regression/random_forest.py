# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 11:39:24 2018

@author: siddhant
"""
#Random Forest Regression

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

#Fitting Random forest regression to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

#predicting results for 6.5 position using Random forest regression
y_pred = regressor.predict(6.5)

#trying with 100 trees
from sklearn.ensemble import RandomForestRegressor
regressor1 = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor1.fit(X, y)
y_pred1 = regressor1.predict(6.5)

#trying with 300 trees
from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor2.fit(X, y)
y_pred2 = regressor2.predict(6.5)

"""#Visualising regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or bluff: Polynomial Regression')
plt.xlabel('pos')
plt.ylabel('salary')
plt.show()"""
#above plot removed because of same problem as decision tree(non-continuous)

#Visualising regression results (for higher resolution and smoother curve)
Xgrid = np.arange(min(X), max(X), 0.01)
Xgrid = Xgrid.reshape((len(Xgrid),1))
plt.scatter(X, y, color = 'red')
plt.plot(Xgrid, regressor.predict(Xgrid), color = 'blue')
plt.title('Truth or bluff: Random forest regression')
plt.xlabel('pos')
plt.ylabel('salary')
plt.show()

