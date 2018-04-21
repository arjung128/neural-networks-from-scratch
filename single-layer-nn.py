
import numpy as np

# If under y = x, 0, else, 1.
data = [[0.5, 3, 1],
        [1.4, 8, 1],
        [7, 3.6, 0],
        [34, 2, 0],
        [6, 6, 1],
        [2, 1.9, 0],
        [3.1, 3.7, 1]]

mystery_data_point = [33, 1]

# As 2 inputs and 1 output, model: y = (w_1)(x_1) + (w_2)(x_2) + b
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

# Define sigmoid and derivative of sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sig_prime(x):
    return (sigmoid(x) * (1 - sigmoid(x)))

# Optimize weights
learning_rate = 0.01
total_cost = 0

for i in range(10000):

    for j in range(len(data)):
        data_point = data[j]

        # Forward-prop: calculate cost
        z = w1 * data_point[0] + w2 * data_point[1] + b
        pred = sigmoid(z)

        target = data_point[2]
        cost = np.square(pred - target)
        total_cost += cost

        # Back-prop: calculate derivatives
        dcost_dpred = 2 * (pred - target)
        dpred_dz = sig_prime(z)
        dz_dw1 = data_point[0]
        dz_dw2 = data_point[1]
        dz_db = 1

        # Chain rule to get dCost / dParameter
        dcost_dw1 = dcost_dpred * dpred_dz * dz_dw1
        dcost_dw2 = dcost_dpred * dpred_dz * dz_dw2
        dcost_db = dcost_dpred * dpred_dz * dz_db

        # Update parameters
        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        b = b - learning_rate * dcost_db

    if(i % 1000 == 0):
        print(total_cost)

    total_cost = 0

print(sigmoid(w1 * mystery_data_point[0] + w2 * mystery_data_point[1]))
