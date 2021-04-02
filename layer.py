from abc import ABC, abstractmethod


class Layer:
    def __init__(self, prev_layer):
        self.prev_layer = prev_layer
        self.next_layer = None
        self.output = None

    @abstractmethod
    def forward(self, inputs, y_true=None):
        pass

    @abstractmethod
    def backward(self, dvalues, y_true=None):
        pass

