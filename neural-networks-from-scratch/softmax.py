import numpy as np
from layer import Layer

# ALthough this is an activation layer,
# it cannot subclass Activation

class Softmax(Layer):
    def forward(self, input):
        e_vals = np.exp(input - np.max(input, axis=-1, keepdims=True))
        self.output = e_vals / np.sum(e_vals, axis=-1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = self.output.shape[0]
        
        # Create the jacobian matrix for softmax with a batch of outputs
        jacobian = np.diagflat(self.output) - np.outer(self.output, self.output)
        
        # Now compute the gradient of the loss w.r.t. input
        return np.dot(jacobian, output_gradient)
        
        # old way as per TIC
        #n = np.size(self.output)
        #tmp = np.tile(self.output, n)
        #return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)