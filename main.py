from fc import FullyConnected
from openslr import OpenSLR

slr = OpenSLR()
train_x,train_y = mnist.get_train_data()
test_x,test_y = mnist.get_test_data()

fc = FullyConnected(train_x, train_y, test_x, test_y)
fc.train()

