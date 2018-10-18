# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 23:58:42 2018

@author: siddhant
"""

#PCA - Implementation

#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %matplotlib inline

# Dataset contains 762 non authentic notes and 610 authentic notes
columns = ["var","skewness","curtosis","entropy","class"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/00267/\
data_banknote_authentication.txt",index_col=False, names = columns)

data_description = {}
for i in df.columns:
    data_description[i] = df[i].describe()

f, ax = plt.subplots(1, 4, figsize=(10,3))
vis1 = sns.distplot(df["var"],bins=10, ax= ax[0])
vis2 = sns.distplot(df["skewness"],bins=10, ax=ax[1])
vis3 = sns.distplot(df["curtosis"],bins=10, ax= ax[2])
vis4 = sns.distplot(df["entropy"],bins=10, ax=ax[3])

sns.pairplot(df, hue="class")

X = df.iloc[:,0:4].values
y = df.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

#Obtain Eigen vectors and eigen values (.T is for transpose)

#from covariance matrix (type 1)
mean_vec = np.mean(X_scaled, axis=0)
cov_mat = (((X_scaled - mean_vec).T).dot(X_scaled-mean_vec)) / (X_scaled.shape[0]-1)

#eigen decomposition on covariance matrix
e_vals1, e_vecs1 = np.linalg.eig(cov_mat)

#from correlation matrix (type 2)
corr_mat = np.corrcoef(X_scaled.T)
e_vals, e_vecs = np.linalg.eig(corr_mat)

#SVD(type 3)
u, s, v = np.linalg.svd(X_scaled.T)

#(eigenvalue, eigenvector) tuples
e_pairs = [(np.abs(e_vals[i]), e_vecs[:,i]) for i in range(len(e_vals))]

e_pairs.sort()
e_pairs.reverse()

for i in e_pairs:
    print(i[0])
    
total = sum(e_vals)
total

var_exp = [(i/total)*100 for i in sorted(e_vals, reverse=True)]

#Cummulative sum
np.cumsum(var_exp)

#horizontal stack
matrix_w = np.hstack((e_pairs[0][1].reshape(4,1),
                      e_pairs[1][1].reshape(4,1)))
#print('Matrix W:\n', matrix_w)
X_new = X_scaled.dot(matrix_w)

#projection ofnew feature space
df["PC1"] = X_new[:,0]
df["PC2"] = X_new[:,1]

#scatter_kws is marker size and size is graph size or window size
sns.lmplot(data = df[["PC1","PC2","class"]], x = "PC1", y = "PC2",fit_reg=False,hue = "class" ,\
           size = 6, aspect=1.5, scatter_kws = {'s':50}, )

sns.pairplot(df[["PC1","PC2","class"]], hue="class")

#implementing PCA using Sci-kit learn
from sklearn.decomposition import PCA
model = PCA(n_components = 2)
X_new2 = model.fit_transform(X_scaled)

df["PC3"] = X_new2[:, 0]
df["PC4"] = X_new2[:, 1]







