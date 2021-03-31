from layer import Layer


class InputLayer(Layer):
    def forward(self, inputs, y_true=None):
        self.output = inputs
        return self.output

    def backward(self, dvalues, y_true=None):
        pass

    def __init__(self, n_inputs):
        super().__init__(n_inputs, n_inputs)


