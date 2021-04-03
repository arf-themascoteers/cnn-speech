from activation_relu import ActivationReLU
from input_layer import InputLayer
from layer_cnn import LayerCNN
from layer_dense import LayerDense
from layer_flatten import LayerFlatten
from layer_maxpool import LayerMaxPool
from optimizer_adam import OptimizerAdam
from output_layer import OutputLayer
import numpy as np


class CNN:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.test_x = test_x

        self.labels = np.unique(train_y)
        self.labels.sort()

        self.train_y = np.array([self.labels.tolist().index(i) for i in train_y])
        self.test_y = np.array([self.labels.tolist().index(i) for i in test_y])

        self.shuffle_train_data()

        _,height, width = test_x.shape

        # Modifying the data shape for CNN layer
        self.train_x = np.expand_dims(self.train_x, axis=1)
        self.test_x = np.expand_dims(self.test_x, axis=1)

        self.optimizer = OptimizerAdam(learning_rate=0.05, decay=0.0001)

        self.input_layer = InputLayer()
        self.output_layer = OutputLayer(self.input_layer)
        self.input_layer.next_layer = self.output_layer

        cnn_layer1 = self.add_layer(LayerCNN(self.input_layer))
        relu1 = self.add_layer(ActivationReLU(cnn_layer1))
        maxpool1 = self.add_layer(LayerMaxPool(relu1))

        flatten = self.add_layer(LayerFlatten(maxpool1))
        n_neurons = 10240
        dense = self.add_layer(LayerDense(flatten, n_neurons, len(self.labels)))

        self.accuracy = 0
        self.loss = 0
        self.N_EPOCH = 5
        self.BATCH_SIZE = 20

    def get_count_output(self, cnn, maxpool, height, width):
        return (height // maxpool.pool) * (width // maxpool.pool) * cnn.N_FILTERS

    def add_layer(self, layer):
        source_layer = self.output_layer.prev_layer
        source_layer.next_layer = layer
        layer.prev_layer = source_layer

        layer.next_layer = self.output_layer
        self.output_layer.prev_layer = layer
        return layer

    def forward(self, input, output):
        layer = self.input_layer
        while layer is not None:
            input = layer.forward(input, output)
            layer = layer.next_layer

        predictions = np.argmax(self.output_layer.output, axis=1)
        self.accuracy = np.mean(predictions == output)
        self.loss = self.output_layer.calculated_loss

        return self.output_layer.output

    def backward(self, dvalues, y_true=None):
        layer = self.output_layer
        while layer != self.input_layer:
            layer.backward(dvalues, y_true)
            dvalues = layer.dinputs
            layer = layer.prev_layer

    def forward_backward(self, input, output):
        dvalues = self.forward(input, output)
        self.backward(dvalues, output)

    def train(self):
        print(f"Starting training with {self.train_y.shape[0]} test data. " +
              f"{self.N_EPOCH} epoch with {self.BATCH_SIZE} samples per batch")
        for epoch in range(self.N_EPOCH):
            self.train_epoch(epoch)

    def train_epoch(self, epoch):
        iterations = self.train_y.shape[0] // self.BATCH_SIZE
        if self.train_y.shape[0] % self.BATCH_SIZE != 0:
            iterations = iterations + 1

        for i in range(iterations):
            self.train_iteration(epoch, i)

        self.print_epoch_summary(epoch)

    def train_iteration(self, epoch, iteration):
        start = iteration * self.BATCH_SIZE
        end = iteration * self.BATCH_SIZE + self.BATCH_SIZE
        self.forward_backward(self.train_x[start: end], self.train_y[start:end])
        self.optimizer.optimise(self)
        self.print_iteration_summary(epoch, iteration)

    def print_epoch_summary(self, epoch):
        print(f'******After End of epoch: {epoch}*****\n' +
              f'acc: {self.accuracy:.3f} , ' +
              f'loss: {self.loss:.3f} , ' +
              f'lr: {self.optimizer.current_learning_rate:.3f} ')

    def print_iteration_summary(self, epoch, iteration):
        print(f"Epoch#{epoch} Batch#{iteration}")
        print(f'epoch: {epoch} , ' +
              f'acc: {self.accuracy:.3f} , ' +
              f'loss: {self.loss:.3f} , ' +
              f'lr: {self.optimizer.current_learning_rate:.3f} ')

    def shuffle_train_data(self):
        length = self.train_y.size
        indexes = np.array(range(length))
        np.random.shuffle(indexes)
        self.train_x = self.train_x[indexes]
        self.train_y = self.train_y[indexes]

    def predict(self, x, y):
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        self.forward(x, y)
        np.argmax(self.output_layer.output, axis=1)
        return self.accuracy, self.loss

    def test(self):
        accuracy_loss = [self.predict(i, j) for i, j in zip(self.test_x, self.test_y)]
        accuracy = [i[0] for i in accuracy_loss]
        loss = [i[1] for i in accuracy_loss]
        return np.mean(accuracy), np.mean(loss)
