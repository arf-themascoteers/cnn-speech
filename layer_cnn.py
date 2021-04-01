import numpy as np

from layer import Layer


class LayerCNN(Layer):
    def __init__(self, n_inputs, n_neurons, n_filters):
        super().__init__(n_inputs, n_neurons)
        self.n_filters = n_filters
        self.stride = 3
        self.filters = np.random.randn(self.n_filters, self.stride) / (self.n_filters)

    def iterate_regions(self, timeseries):
        _, width = timeseries.shape

        for i in range(width - self.stride + 1):
            im_region = image[i:(i + 3), j:(j + 3)]
            yield im_region, i, j


    def forward(self, inputs, y_true=None):
        self.inputs = inputs
        _, width = inputs.shape
        output = np.zeros(width - self.stride+1, self.n_filters)

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues, y_true=None):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

