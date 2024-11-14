import numpy as np
from layer import Layer
class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        # Derivative is 1 where input > 0, else 0
        return output_gradient * (self.input > 0)