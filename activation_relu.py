import numpy as np

from layer import Layer


class ActivationReLU(Layer):
    def __init__(self, n_inputs):
        super().__init__(n_inputs, n_inputs)

    def forward(self, inputs, y_true=None):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues, y_true=None):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
