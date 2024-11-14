import numpy as np
from layer import Layer
from scipy import signal
# Note: below he defines the output shape as the easy formula which is 
#       highly dependent on "valid" passed to the correlation function

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, 
                             input_height - kernel_size + 1, 
                             input_width - kernel_size + 1)
        self.kernels_shape = (depth, 
                              input_depth, 
                              kernel_size, 
                              kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases  = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth): # output depth
            for j in range(self.input_depth): # input depth
                # warning the correlation is not communative... orders matter!!! X star K
                self.output[i] += signal.correlate2d(self.input[j], 
                                                       self.kernels[i,j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernel_gradient = np.zeros(self.kernels.shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernel_gradient[i, j] = signal.correlate2d(self.input[j], 
                                                             output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], 
                                                       self.kernels[i,j], "full")

        self.kernels -= learning_rate * kernel_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient           
        