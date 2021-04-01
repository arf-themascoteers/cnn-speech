from input_layer import InputLayer
from layer_cnn import LayerCNN
from layer_dense import LayerDense
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

        self.input_layer = InputLayer(self.train_x.shape[1])
        self.output_layer = OutputLayer(len(self.labels), self.input_layer)
        self.input_layer.prev_layer = self.output_layer
        self.accuracy = 0
        self.loss = 0
        N_FILTER = 3
        self.add_layer(LayerCNN(self.train_x.shape[1], N_FILTER))
        self.add_layer(LayerMaxPool(self.train_x.shape[1], N_FILTER))
        self.add_layer(LayerDense(672,32))
        self.add_layer(LayerDense(32,len(self.labels)))
        self.optimizer = OptimizerAdam(learning_rate=0.05, decay=5e-7)


    def add_layer(self, layer):
        source_layer = self.output_layer.prev_layer

        source_layer.next_layer = layer
        layer.prev_layer = source_layer

        layer.next_layer = self.output_layer
        self.output_layer.prev_layer = layer

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

    def print_forward(self):
        layer = self.input_layer
        print("Machine")
        print("=======")
        while layer is not None:
            print(f"{layer.n_neurons} - {type(layer).__name__}")
            layer = layer.next_layer

    def train(self):
        for epoch in range(400):
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
        new_x = np.zeros((len(self.train_x),len(self.train_x[0])))
        for old_index,index in enumerate(indexes):
            new_x[old_index] = self.train_x[index]
        self.train_x = new_x
        self.train_y = self.train_y[indexes]


    def predict(self, x, y):
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        self.forward(x,y)
        predicted = np.argmax(self.output_layer.output, axis=1)
        return self.accuracy

    def test(self):
        accuracy = [self.predict(i,j) for i,j in zip(self.test_x, self.test_y)]
        return np.mean(accuracy)
