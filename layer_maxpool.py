import numpy as np

from layer import Layer


class LayerMaxPool(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self.inputs = None
        self.pool = 2

    def iterate_regions(self):
        n_items, n_windows, height, width = self.inputs.shape
        new_height = height//self.pool
        new_width = width//self.pool

        for item in range(n_items):
            for window in range(n_windows):
                for height_index in range(new_height):
                    for width_index in range(new_width):
                        height_start_index = height_index *self.pool
                        height_end_index = height_start_index + self.pool

                        width_start_index = width_index *self.pool
                        width_end_index = width_start_index + self.pool

                        image_region = self.inputs[item,
                                                   window,
                                                   height_start_index : height_end_index,
                                                   width_start_index : width_end_index]
                        yield image_region, item, window, height_index, width_index

    def forward(self, inputs, y_true=None):
        self.inputs = inputs
        n_items, n_windows, height, width = self.inputs.shape
        self.output = np.zeros((n_items, n_windows, height//self.pool, width//self.pool))
        for image_region, item, window, height_index, width_index in self.iterate_regions():
            self.output[item, window, height_index, width_index] = np.amax(image_region, axis=(0, 1))
        return self.output

    def backward(self, dvalues, y_true=None):
        self.dinputs = np.zeros(self.inputs.shape)

        for image_region, item, window, height_index, width_index in self.iterate_regions():
            max_index = np.argmax(image_region)
            (x, y) = np.unravel_index(max_index, (self.pool, self.pool))
            update_height_index = height_index + x
            update_width_index = width_index + y
            update_value = dvalues[item, window, height_index // self.pool, width_index // self.pool]
            self.dinputs[item, window, update_height_index, update_width_index] = update_value

        return self.dinputs

