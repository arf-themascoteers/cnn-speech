from activation_softmax import ActivationSoftmax
from layer import Layer
from loss_cce import LossCategoricalCrossentropy
import numpy as np


class OutputLayer(Layer):
    def __init__(self, n_inputs, prev_layer):
        super().__init__(n_inputs, 1, prev_layer)
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
        if len (y_true.shape) == 2 :
            y_true = np.argmax(y_true, axis = 1 )
        self.dinputs = dvalues.copy()
        self.dinputs[ range (samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

