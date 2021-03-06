from activation_relu import ActivationReLU
from input_layer import InputLayer
from layer_dense import LayerDense
from optimizer_adam import OptimizerAdam
from output_layer import OutputLayer
import numpy as np

class FullyConnected:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.test_x = test_x

        self.labels = np.unique(train_y)
        self.labels.sort()

        self.train_y = np.array([self.labels.tolist().index(i) for i in train_y])
        self.test_y = np.array([self.labels.tolist().index(i) for i in test_y])

        self.shuffle_train_data()

        self.input_layer = InputLayer()
        self.output_layer = OutputLayer(self.input_layer)
        self.input_layer.next_layer = self.output_layer
        layer = self.add_layer(LayerDense(self.input_layer, self.train_x.shape[1], 1024))
        layer = self.add_layer(ActivationReLU(layer))
        layer = self.add_layer(LayerDense(layer, 1024, len(self.labels)))
        self.optimizer = OptimizerAdam(learning_rate=0.05, decay=0.0001)

        self.accuracy = 0
        self.loss = 0
        self.N_EPOCH = 401


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
        for epoch in range(self.N_EPOCH):
            self.forward_backward(self.train_x, self.train_y)
            self.optimizer.optimise(self)

            if not epoch % 100:
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
        self.forward(x,y)
        np.argmax(self.output_layer.output, axis=1)
        return self.accuracy, self.loss

    def test(self):
        accuracy_loss = [self.predict(i,j) for i,j in zip(self.test_x, self.test_y)]
        accuracy = [i[0] for i in accuracy_loss]
        loss = [i[1] for i in accuracy_loss]
        return np.mean(accuracy), np.mean(loss)
