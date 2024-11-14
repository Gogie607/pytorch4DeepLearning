#  create Mean Squared error and its derivative
import numpy as np
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))
#
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

"""
# Create the Binary cross entropy function and its derivative
# Note that y_true and y_pred are vectors and not scalars
 I am using redundant parenthesis in the functions for clarity 
"""
# E = (-1/n)sum[i=1..n] (Y*i log(yi) + (1-Y*i)log(1-Yi))

def binary_cross_entropy(y_true, y_pred):
    # notice the log(y_pred) and the log(1 - y_pred)
    # neither of these can be negative since log(-num)  is illegal
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

# pE/pYi = (1/n)((1-Y*i)/(1-Yi) - (Y*i/Yi)) 
def binary_cross_entropy_prime(y_true, y_pred):
    return  ((1 - y_true) / (1 - y_pred)) - (y_true / y_pred) / np.size(y_true)

# Loss_{CE} = -\sum^c_{i = 1}p(x_i) \cdot log(\hat p(x_i))
def cross_entropy(y_true, y_pred):
    # Small value to avoid log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)

    return -np.sum(y_true * np.log(y_pred))
# partial Loss / partial pred = -true/pred   but can be simplified
def cross_entropy_prime (y_true, y_pred):
    return y_pred - y_true