import numpy as np

from layer import Layer


class LayerCNN(Layer):
    def __init__(self, prev_layer, input_windows):
        super().__init__(prev_layer)
        self.stride = 3
        self.N_FILTERS = 4
        self.input_windows = input_windows
        self.weights = np.random.randn(self.N_FILTERS, self.input_windows, self.stride, self.stride) / (self.stride * self.stride)
        self.dweights = None
        self.inputs = None

    def padding(self, input):
        n_items, n_windows, height, width = input.shape
        new_height = height + self.stride - 1
        new_width = width + self.stride - 1
        out = np.zeros((n_items, n_windows, new_height, new_width))
        pad = self.stride // 2
        out[:, :, pad: pad + height, pad: pad + width] = input
        return out

    def iterate_regions(self):
        n_items, n_windows, height, width = self.inputs.shape

        for item_index in range(n_items):
            for kernel_index,kernel in enumerate(self.weights):
                for height_index in range(height - self.stride + 1):
                    for width_index in range(width - self.stride + 1):
                        image_region = self.inputs[
                                       item_index,
                                       :,
                                       height_index: (height_index + self.stride),
                                       width_index: (width_index + self.stride)]
                        kernel = self.weights[kernel_index]
                        yield item_index, image_region, height_index, width_index, kernel_index, kernel

    def forward(self, inputs, y_true=None):
        self.inputs = inputs
        n_items, n_windows, height, width = inputs.shape
        self.inputs = self.padding(self.inputs)
        self.output = np.zeros((n_items, self.N_FILTERS, height, width))

        for item_index, image_region, height_index, width_index, kernel_index, kernel in self.iterate_regions():
            self.output[item_index, kernel_index, height_index, width_index] = np.sum(image_region * kernel)

        return self.output

    def backward(self, dvalues, y_true=None):
        self.dweights = np.zeros_like(self.weights)
        self.dinputs = np.zeros_like(self.inputs)

        for item_index, image_region, height_index, width_index, kernel_index, kernel in self.iterate_regions():
            self.dweights[kernel_index] += dvalues[item_index, kernel_index, height_index, width_index] * image_region
            self.dinputs[item_index,
                         :,
                         height_index: (height_index + self.stride),
                         width_index: (width_index + self.stride)
                        ] += dvalues[item_index, :, height_index, width_index] * kernel
