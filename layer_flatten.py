import numpy as np

from layer import Layer


class LayerFlatten(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self.n_filters = None
        self.n_items = None
        self.height = None
        self.width = None
        self.inputs = None
        self.output = None

    def forward(self, inputs, y_true=None):
        self.inputs = inputs
        self.n_items, self.n_filters, self.height, self.width = self.inputs.shape
        self.output = self.inputs.reshape(self.n_items, -1)
        return self.output

    def backward(self, dvalues, y_true=None):
        self.dvalues = dvalues.reshape(self.n_items, self.n_filters, self.height, self.width)


