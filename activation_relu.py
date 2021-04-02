import numpy as np

from layer import Layer


class ActivationReLU(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer)

    def forward(self, inputs, y_true=None):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues, y_true=None):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
