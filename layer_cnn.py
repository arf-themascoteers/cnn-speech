import numpy as np

from layer import Layer


class LayerCNN(Layer):
    def __init__(self, n_inputs, n_filters):
        super().__init__(n_inputs, n_inputs)
        self.n_filters = n_filters
        self.stride = 3
        self.filters = np.random.randn(self.n_filters, self.stride, self.stride) / (self.n_filters)

    def iterate_regions(self, frame):
        height, width = frame.shape

        for i in range(height - self.stride + 1):
            if i + self.stride < height:
                for j in range(width - self.stride + 1):
                    if j + self.stride < width:
                        im_region = frame[i:(i + self.stride), j:(j + self.stride)]
                        yield im_region, i, j


    def forward(self, inputs, y_true=None):
        self.inputs = inputs
        n_items, height, width = inputs.shape
        self.output = np.zeros(n_items, height - self.stride+1, self.n_filters)
        for filter in range(self.n_filters):
            for im_region, i, j in self.iterate_regions(inputs):
                self.output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return self.output

    def backward(self, dvalues, y_true=None):
        n_filters, height, width = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)
        self.dinputs = np.dot(dvalues, self.weights.T)


