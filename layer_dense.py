import numpy as np

from layer import Layer


class LayerDense(Layer):
    def __init__(self, prev_layer,n_inputs, n_neurons):
        super().__init__(prev_layer)
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.inputs = None

    def forward(self, inputs, y_true=None):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues, y_true=None):
        self.dweights = np.dot(self.inputs.T, dvalues) / self.inputs.shape[0]
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) / self.inputs.shape[0]
        self.dinputs = np.dot(dvalues, self.weights.T) / self.inputs.shape[0]
