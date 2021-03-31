from abc import ABC, abstractmethod


class Layer:
    def __init__(self, n_inputs, n_neurons, prev_layer = None, next_layer = None):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.output = None

    @abstractmethod
    def forward(self, inputs, y_true=None):
        pass

    @abstractmethod
    def backward(self, dvalues, y_true=None):
        pass

