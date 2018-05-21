import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import sklearn.metrics
import pylab

# Generate Dataset
examples = 2
features = 3
D = (npr.randn(examples, features), npr.randn(examples))

# input data
    # print(D[0]) # a list of arrays with [examples] # of elements and [features] # of numbers in each array

# output data (correct label)
    # print(D[1]) # a list of numbers with [examples] # of elements

# Specify the network
layer1_units = 10
layer2_units = 1
w1 = npr.rand(features, layer1_units)

# print(w1)   # a list of arrays with [features] # of elements and [layer1_units] # of numbers in each array
              # Note: these are rand, not randn as above, meaning these are all positive.

              # when np.dot(x, w1) is performed you get [examples, layer1_units] as the output

# wX = np.dot(D[0], w1)
# print(wX.shape)

b1 = npr.rand(layer1_units)
# Why isn't this rand(examples, layer1_units)? why only 1 row of biases?
# The above should NOT be rand(examples, layer1_units) because we want there to be as many biases
# as there are nodes/neurons in the 1st layer. the examples dimension is just like batch size here,
# and we want [layer1_units] many biases, not [layer1_units * batch size]
# rand(examples, layer1_units) would create [examples, layer1_units] many biases, which we do not want

# print(wX)
# print(b1)
# wXb = wX + b1
# Despite the fact that wX is [example,s layer1_units] in shape and b1 is [layer1_units] in shape,
# the biases are added to each corresponding element in the wX matrix correctly.

# print(wXb.shape)
# print(wXb)

w2 = npr.rand(layer1_units, layer2_units)
# so that when you do np.dot(wX + b, w2) you get [examples, layer2_units] as the output

b2 = 0.0
# why is this initialized to 0?

theta = (w1, b1, w2, b2)
# a dictionary of all weights and biases

# print(theta[0])   # same as print(w1)

# Define the loss function
def squared_loss(y, y_hat):
    return np.dot((y - y_hat),(y - y_hat))
    # With this, RMSE is regularly reduced by 8 orders of magnitude.

    # ABSOLUTE VALUE
    # if (y - y_hat) > 0:
    #     return (y - y_hat)
    # else:
    #     return (y_hat - y)
    # With this, in the best case, RMSE is reduced by 1 order of magnitude.

# Wrapper around the Neural Network
def neural_network(x, theta):
    w1, b1, w2, b2 = theta
    return np.tanh(np.dot((np.tanh(np.dot(x,w1) + b1)), w2) + b2)
# neural network math with activation function involved

# Wrapper around the objective function to be optimised
def objective(theta, idx):
    return squared_loss(D[1][idx], neural_network(D[0][idx], theta))
# returns squared_loss for given example (example is specified with idx)

# Back-prop
def update_theta(theta, delta, alpha):
    w1, b1, w2, b2 = theta
    w1_delta, b1_delta, w2_delta, b2_delta = delta
    w1_new = w1 - alpha * w1_delta
    b1_new = b1 - alpha * b1_delta
    w2_new = w2 - alpha * w2_delta
    b2_new = b2 - alpha * b2_delta
    new_theta = (w1_new,b1_new,w2_new,b2_new)
    return new_theta

# Compute Gradient
grad_objective = grad(objective)


# Train the Neural Network
epochs = 1000

print("RMSE before training:", sklearn.metrics.mean_squared_error(D[1], neural_network(D[0], theta)))

for i in range(epochs):
    for j in range(examples):
        delta = grad_objective(theta, j)
        theta = update_theta(theta,delta, 0.01)

print("RMSE after training:", sklearn.metrics.mean_squared_error(D[1], neural_network(D[0], theta)))
