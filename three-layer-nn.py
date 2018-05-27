import pandas as pd
import numpy as np
import sklearn

import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('wine2.csv')

# initialize parameters
w1 = np.random.randn(13)
w1 = np.expand_dims(w1, axis=0) # (1, 13)
print(w1.shape)
b1 = np.random.randn(1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

for i in range(60):

    yhat = df.values[i][0]
    # print(yhat)
    x = df.values[i][1:]
    x = np.expand_dims(x, axis=1) # (13, 1)

    print(i, w1.shape)
    y1 = np.dot(w1, x) + b1
    print(i, w1.shape)
    # print(w1.shape)
    # print(y1)

    z1 = sigmoid(y1)
    # print(z1)

    cost = np.square(z1 - yhat)
    # print(i, cost)

    dcost_dz1 = 2 * (z1 - yhat)
    dz1_dy1 = (sigmoid(y1) * (1 - sigmoid(y1)))
    dy1_dw1 = x
    dy1_db1 = 1

    dcost_dw1 = dcost_dz1 * dz1_dy1 * dy1_dw1
    dcost_db1 = dcost_dz1 * dz1_dy1 * dy1_db1

    print(i, w1.shape)
    w1 = w1 - 0.01 * dcost_dw1
    print(i, w1.shape)
    b1 = b1 - 0.01 * dcost_db1

for j in range(1):
    yhat = df.values[100 + j][0]
    x = df.values[100 + j][1:]
    x = np.expand_dims(x, axis=1) # (13, 1)
    print(x.shape)
    print(w1.shape)
    y1 = np.dot(w1, x) + b1
    print(y1.shape)
    z1 = sigmoid(y1)
    print(z1.shape)
    print("actual: ", yhat, ", prediction: ", z1)

# training loop
# for i in range(178):
#     x = df.values[i][1:]
