class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self,input):
        # TODO: returns output Y
        pass

    def backward(self, output_gradient, learning_rate):
        """
        params: output_gradient - partial derived loss wrt this layers output
                learning_rate - some constant applied to the output_gradient to adjust the weights
        returns: input_gradient  pE/pX  (which in turn becomes the pE/pY of the previous layer
         *optionaly passing in and returning an optimizer that encapsulates the gradient descent
        """
        # TODO update parameters and return input gradient (pE/pX)
        pass