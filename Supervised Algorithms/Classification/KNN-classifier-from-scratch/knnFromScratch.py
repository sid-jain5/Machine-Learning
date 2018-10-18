# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 09:21:13 2018

@author: siddhant
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train =sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

import math
def knnclassifier(x_test, x_train, y_train, no_of_neighbours):
    
    dist = []
    #Euclidean distance of x_test from all x_train
    for i in range(0,len(x_train)):
        d = (x_train[i][0] - x_test[0]) ** 2
        e = (x_train[i][1] - x_test[1]) ** 2
        f = math.sqrt(d+e)
        dist.append(f)
    #print(len(dist))
    
    dy = sorted(zip(dist, y_train))
    y_sorted = [y for x,y in dy]
    
    neighbours = y_sorted[:no_of_neighbours]
    
    #classset = {}
    
    
    count0 = 0
    count1 = 0
    for k in range(0,5):
        if neighbours[k] == 0:
            count0 = count0 + 1
        else:
            count1 = count1 + 1
            
    if count0 > count1:
        return 0
    else:
        return 1


        
y_pred = []
for j in range (0,len(x_test)):
    yp = knnclassifier(x_test[j], x_train, y_train,5)
    y_pred.append(yp)
    
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(x_train, y_train)

#Predicting the test set result
y_pred2 = classifier.predict(x_test)
