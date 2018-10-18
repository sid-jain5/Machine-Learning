# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 22:28:55 2018

@author: siddhant
"""

#%reset -f

#Apriori

#Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Imporing the Dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

#make list of transactions or list of lists(each row is a transaction) or
#group of items brought together
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

#Training Apriori on the dataset 
from apyori import apriori
"""support = ThisItemPurchased/TotalItemPurchased
lets take products that are purchased atleast 3 to 4 times a day
since data is of 7 days implies item is purchased 3*7 times a week
total items purchased are 7500 implies support = (3*7)/7500 = 0.0028"""

"""confidence value should not be high as it would mean that both items
are purchased a lot and not that they associate well with each other
usually default value is 80% or 0.8 but we will take 20% as it gives 
better results"""

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualising the results
results = list(rules)