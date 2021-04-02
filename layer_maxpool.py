import numpy as np

from layer import Layer


class LayerMaxPool(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self.inputs = None
        self.pool = 2
        self.n_filters = None

    def iterate_regions(self, image):
        height, width = image.shape
        new_height = height//self.pool
        new_width = width//self.pool

        for i in range(new_height):
            for j in range(new_width):
                im_region = image[(i *self.pool):(i *self.pool +self.pool), (j*self.pool):(j*self.pool + self.pool)]
                yield im_region, i, j


    def forward(self, inputs, y_true=None):
        self.inputs = inputs
        n_filters, n_items, height, width = self.inputs.shape
        self.output = np.zeros(n_filters, n_items, height//self.pool, width//self.pool)
        self.n_filters = n_filters
        for filter in range(self.n_filters):
            for item in range(n_items):
                for im_region, i, j in self.iterate_regions(self.inputs[filter][item]):
                    self.output[filter, item, i, j] = np.amax(im_region, axis=(0, 1))

        return self.output

    def backward(self, dvalues, y_true=None):
        n_filters, n_items, height, width = self.inputs.shape
        self.dinputs = np.zeros(self.inputs.shape)

        for filter in range(self.n_filters):
            for item in range(n_items):
                for im_region, i, j in self.iterate_regions(self.inputs[filter][item]):
                    max_index = np.argmax(im_region)
                    (x,y) = np.unravel_index(max_index,(self.pool,self.pool))
                    self.dinputs[filter, i+x, j+y] = dvalues[filter, height//self.pool, width//self.pool]

        return self.dinputs


