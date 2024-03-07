""" This algorithm is a representation of the supervised machine learning algorithm KNearest Neighbors Regression 

the algoritm uses the Xtrain and ytrain to take the average of the k nearest of points y coordinates. 
(k is a number that can be set by the user and is varied)
the average that was calculated was saved into ypred which is used as the algorithm calculated answer
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Data Table Creation
#X = np.array([[1],[4],[5],[2],[6],[9]])
#y = np.array([4,16,20,8,24,36])
X,y = make_regression(n_samples = 100, n_features = 1, noise = 10)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

# Main Algorithm
alg = KNeighborsRegressor(n_neighbors = 3)
alg.fit(Xtrain,ytrain)
ypred = alg.predict(Xtest)

# Graphing
plt.scatter(Xtrain , ytrain)
plt.scatter(Xtest, ypred, marker = "x")
plt.grid()
plt.figure()