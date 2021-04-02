import numpy as np

from layer import Layer


class LayerFlatten(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self.n_filters = None
        self.n_items = None
        self.height = None
        self.width = None

    def forward(self, inputs, y_true=None):
        self.n_filters, self.n_items, self.height, self.width = inputs.shape
        return inputs.reshape(-1, self.height, self.width)

    def backward(self, dvalues, y_true=None):
        return dvalues.reshape(self.n_filters, self.n_items, self.height, self.width)


