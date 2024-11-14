from layer import Layer     # assume class is defined in its own package
import numpy as np
#
class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
#
    def forward(self, input):
        self.input = input
        
        return np.dot(self.weights, self.input) + self.bias
#
    def backward(self, output_gradient, learning_rate):
        weights_gradients = np.dot(output_gradient, self.input.T) # loss dot transpose(X)
        self.weights -= learning_rate * weights_gradients
        self.bias -= learning_rate * output_gradient  # just the loss itself
        return np.dot(self.weights.T, output_gradient) # transpose weights dot loss
    