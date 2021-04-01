import numpy as np

from layer import Layer


class LayerMaxPool(Layer):
    def __init__(self, n_inputs, n_filters):
        super().__init__(n_inputs, n_inputs)
        self.inputs = None
        self.n_filters = n_filters
        self.pool = 2

    def iterate_regions(self, frame):
        height, width = frame.shape
        new_height = height//self.pool
        new_width = width//self.pool

        for i in range(new_height):
            if i + self.stride < height:
                for j in range(new_width):
                    if j + self.stride < width:
                        im_region = frame[(i *self.pool):(i *self.pool +self.pool), (j*self.pool):(j*self.pool + self.pool)]
                        yield im_region, i, j


    def forward(self, inputs, y_true=None):
        self.inputs = inputs
        n_filters, height, width = inputs.shape
        self.output = np.zeros(n_filters, height//self.pool, width//self.pool)
        for filter in range(n_filters):
            for im_region, i, j in self.iterate_regions(self.inputs[filter]):
                self.output[filter, i, j] = np.amax(im_region, axis=(0, 1))

        return self.output

    def backward(self, dvalues, y_true=None):
        n_filters, height, width = self.inputs.shape
        self.dinputs = np.zeros(self.inputs.shape)

        for filter in range(self.n_filters):
            for im_region, i, j in self.iterate_regions(self.inputs):
                max_index = np.argmax(im_region)
                (x,y) = np.unravel_index(max_index,(self.pool,self.pool))
                self.dinputs[filter, i+x, j+y] = dvalues[filter, height//self.pool, width//self.pool]

        return self.dinputs


