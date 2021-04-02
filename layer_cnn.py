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

    def padding(self, input):
        height, width = input.shape
        new_height = height + self.stride - 1
        new_width = width + self.stride - 1
        out = np.zeros((new_height, new_width))
        pad = self.stride // 2
        out[pad: pad + height, pad: pad + width] = input
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
        self.output = np.zeros( (n_items, self.N_FILTERS, height, width) )
        for item in range(n_items):
            self.apply_filter_on_item(self.weights, self.inputs[item], self.output[item])

        return self.output

    def apply_filter_on_item(self, weights, input_image, output_image):
        for kernel in range(self.N_FILTERS):
            self.apply_filter_on_item_for_kernel(weights[kernel], input_image, output_image)

    def apply_filter_on_item_for_kernel(self, kernel, input_image, output_image):
        for im_region, i, j in self.iterate_regions(input_image):
            output_image[i, j] = np.sum(im_region * kernel)


    def backward(self, dvalues, y_true=None):
        self.dweights = np.zeros_like(self.weights)
        self.dinputs = np.zeros_like(self.inputs)

        for item in range(self.inputs.shape[0]):
            self.backward_filter_on_item(self.weights, self.inputs[item], dvalues[item], self.dweights, self.dinputs[item])

    def backward_filter_on_item(self, weights, input_image, back_output_image, dweights, dinputs):
        for kernel in range(self.N_FILTERS):
            self.backward_filter_on_item_for_kernel(weights[kernel], input_image, back_output_image, dweights[kernel], dinputs[kernel])


    def backward_filter_on_item_for_kernel(self, kernel, input_image, back_output_image, dweight, dinput):
        for im_region, i, j in self.iterate_regions(input_image):
            dweight += back_output_image[i, j] * im_region
            dinput += back_output_image * kernel



