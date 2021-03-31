import numpy as np
import image_viewer
from fc import FullyConnected
from mnist import Mnist

mnist = Mnist()
train_x,train_y = mnist.get_train_data()
test_x,test_y = mnist.get_test_data()

fc = FullyConnected(train_x, train_y, test_x, test_y)
fc.train()

