from layer import Layer


class InputLayer(Layer):
    def forward(self, inputs, y_true=None):
        self.output = inputs
        return self.output

    def backward(self, dvalues, y_true=None):
        pass

    def __init__(self):
        super().__init__(None)


