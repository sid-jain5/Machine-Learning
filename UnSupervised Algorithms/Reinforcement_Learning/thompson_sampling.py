# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 18:39:49 2018

@author: siddhant
"""
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#Importing the Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling model
N = 10000
d = 10
ads_selected = []

num_of_reward_1 = [0] * d
num_of_reward_0 = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(num_of_reward_1[i]+1, num_of_reward_0[i]+1)        
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward =dataset.values[n, ad]
    if reward == 1:
        num_of_reward_1[ad] = num_of_reward_1[ad] + 1
    else:
        num_of_reward_0[ad] = num_of_reward_0[ad] + 1
    total_reward = total_reward + reward
    
    
#if you go down in ads selected the ad number will not change much
#in last 100 rounds it remains the same because of the high reward(in this case ad 4)

#Visualising the result
plt.hist(ads_selected)
plt.title("Histogram of ads selection")
plt.xlabel("Ads")
plt.ylabel("Noumber of times each ad was selected")
plt.show()