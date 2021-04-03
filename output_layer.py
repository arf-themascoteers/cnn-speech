from activation_softmax import ActivationSoftmax
from layer import Layer
from loss_cce import LossCategoricalCrossentropy
import numpy as np


class OutputLayer(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()
        self.calculated_loss = 0

    def forward ( self , inputs, y_true=None ):
        self.activation.forward(inputs)
        self.output = self.activation.output
        self.calculated_loss = self.loss.calculate(self.output, y_true)
        return self.output

    def backward ( self , dvalues, y_true=None ):
        samples = len (dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[ range (samples), y_true] -= 1


