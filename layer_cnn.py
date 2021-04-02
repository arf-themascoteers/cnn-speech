import numpy as np

from layer import Layer


class LayerCNN(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self.stride = 3
        self.N_FILTERS = 4
        self.weights = np.random.randn(self.N_FILTERS, self.stride, self.stride) / (self.stride * self.stride)
        self.dweights = None
        self.inputs = None

    def padding(self, inputs):
        samples, height, width = inputs.shape
        new_height = height + self.stride - 1
        new_width = width + self.stride - 1
        out = np.zeros((samples, new_height, new_width))
        pad = self.stride / 2
        out[:, pad: pad + height, pad: pad + width] = inputs
        return out

    def iterate_regions(self, image):
        height, width = image.shape

        for i in range(height - self.stride + 1):
            for j in range(width - self.stride + 1):
                im_region = image[i:(i + self.stride), j:(j + self.stride)]
                yield im_region, i, j

    def forward(self, inputs, y_true=None):
        self.inputs = inputs
        n_items, height, width = inputs.shape
        self.inputs = self.padding(self.inputs)
        self.output = np.zeros( (self.N_FILTERS, n_items, height, width) )
        for kernel in range(self.N_FILTERS):
            self.apply_filter(self.weights[kernel], self.inputs, self.output[kernel])

        return self.output

    def apply_filter(self, kernel, inputs, output):
        for item in range(inputs.shape[0]):
            self.apply_filter_for_item(kernel, inputs[item], output[item])

    def apply_filter_for_item(self, kernel, input_image, output_image):
        for im_region, i, j in self.iterate_regions(input_image):
            output_image[i, j] = np.sum(im_region * kernel)

    def backward(self, dvalues, y_true=None):
        n_items, height, width = self.inputs.shape

        self.dweights = np.zeros_like(self.weights)
        self.dinputs = np.zeros_like(self.inputs)

        for kernel in range(self.N_FILTERS):
            self.backward_filter(self.weights[kernel], self.inputs, dvalues[kernel], self.dweights[kernel], self.dinputs)

    def backward_filter(self, kernel, inputs, dvalue, dweight, dinputs):
        for item in range(inputs.shape[0]):
            self.backward_filter_for_item(kernel, inputs[item], dvalue[item], dweight, dinputs[item])

    def backward_filter_for_item(self, kernel, input_image, back_output_image, dweight, dinput):
        for im_region, i, j in self.iterate_regions(input_image):
            dweight += back_output_image[i, j] * im_region
            dinput += back_output_image * kernel
