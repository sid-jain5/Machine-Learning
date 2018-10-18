# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:06:06 2018

@author: siddhant
"""
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#Importing the Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
N = 10000
d = 10
ads_selected = []

num_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if(num_of_selections[i] > 0):
            avg_reward = sum_of_rewards[i] / num_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / num_of_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400 #10 to the power 400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    num_of_selections[ad]+=1
    reward =dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
    
#if you go down in ads selected the ad number will not change much
#in last 100 rounds it remains the same because of the high reward(in this case ad 4)

#Visualising the result
plt.hist(ads_selected)
plt.title("Histogram of ads selection")
plt.xlabel("Ads")
plt.ylabel("Noumber of times each ad was selected")
plt.show()

